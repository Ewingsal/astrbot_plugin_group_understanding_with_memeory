from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from astrbot_plugin_group_digest.services.models import (
    MessageRecord,
    SlangExplanationRecord,
    TopicSliceRecord,
)
from astrbot_plugin_group_digest.services.semantic_input_builder import SemanticInputBuilder
from astrbot_plugin_group_digest.services.slang_candidate_miner import SlangCandidate, SlangCandidateMiner
from astrbot_plugin_group_digest.services.slang_interpretation_service import SlangInterpretationService
from astrbot_plugin_group_digest.services.slang_store import SlangStore


def _run(coro):
    return asyncio.run(coro)


def _slice(
    *,
    date_label: str,
    topic_id: str,
    core_text: str,
    start_ts: int,
    end_ts: int,
) -> TopicSliceRecord:
    return TopicSliceRecord(
        group_id="group_1001",
        date_label=date_label,
        topic_id=topic_id,
        start_ts=start_ts,
        end_ts=end_ts,
        message_count=4,
        participants=["Alice(u1)", "Bob(u2)"],
        core_text=core_text,
    )


class _Resp:
    def __init__(self, text: str):
        self.completion_text = text


class _StubContext:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.llm_calls = 0

    async def llm_generate(self, chat_provider_id: str, prompt: str):
        _ = (chat_provider_id, prompt)
        self.llm_calls += 1
        if not self.responses:
            raise RuntimeError("no mock llm response")
        return _Resp(self.responses.pop(0))


class _StubLLMAnalysisService:
    async def resolve_provider_id(self, *, context, event, configured_provider_id):
        _ = (context, event, configured_provider_id)
        return "test_provider", "session", ""


class _StubEmbeddingBackend:
    async def embed_text(self, text: str):
        _ = text
        return [0.1, 0.2]


class _StubEmbeddingStore:
    def __init__(self, rows: list[dict]):
        self.rows = list(rows)
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


class _EmptyTopicManager:
    def collect_slice_contexts(self, **kwargs):
        _ = kwargs
        return []

    def get_day_topics_snapshot(self, **kwargs):
        _ = kwargs
        return []


def test_slang_candidate_miner_discovers_repeated_term() -> None:
    miner = SlangCandidateMiner(
        min_term_frequency=2,
        min_slice_coverage=2,
        max_candidates=10,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())
    current_day = [
        _slice(
            date_label="2026-03-22",
            topic_id="20260322_0001",
            core_text="今晚继续龙王局，先把回滚预案过一遍",
            start_ts=base_ts,
            end_ts=base_ts + 300,
        )
    ]
    recent = [
        _slice(
            date_label="2026-03-21",
            topic_id="20260321_0001",
            core_text="昨天龙王局聊的是部署窗口和值守安排",
            start_ts=base_ts - 86400,
            end_ts=base_ts - 86100,
        ),
        _slice(
            date_label="2026-03-20",
            topic_id="20260320_0001",
            core_text="前天也提到龙王局，大家默认是紧急讨论模式",
            start_ts=base_ts - 2 * 86400,
            end_ts=base_ts - 2 * 86400 + 300,
        ),
    ]

    candidates = miner.mine_candidates(
        current_day_slices=current_day,
        recent_slices=recent,
    )
    terms = {item.term for item in candidates}
    assert "龙王局" in terms


def test_slang_candidate_miner_avoids_common_terms() -> None:
    miner = SlangCandidateMiner(
        min_term_frequency=2,
        min_slice_coverage=2,
        max_candidates=10,
    )
    base_ts = int(datetime(2026, 3, 22, 10, 0, 0).timestamp())
    rows = [
        _slice(
            date_label="2026-03-22",
            topic_id="20260322_0001",
            core_text="今天我们可以讨论这个事情",
            start_ts=base_ts,
            end_ts=base_ts + 60,
        ),
        _slice(
            date_label="2026-03-21",
            topic_id="20260321_0001",
            core_text="今天我们可以讨论那个事情",
            start_ts=base_ts - 86400,
            end_ts=base_ts - 86340,
        ),
    ]

    candidates = miner.mine_candidates(
        current_day_slices=[rows[0]],
        recent_slices=rows,
    )
    assert all(item.term not in {"今天", "我们", "可以", "讨论"} for item in candidates)


def test_slang_interpretation_service_retrieves_context_and_infers(tmp_path: Path) -> None:
    slang_store = SlangStore(tmp_path / "slang")
    embedding_store = _StubEmbeddingStore(
        rows=[
            {
                "object_type": "topic_slice",
                "group_id": "group_1001",
                "date_label": "2026-03-21",
                "topic_id": "20260321_0001",
                "core_text": "龙王局一般表示临时高强度任务讨论",
            },
            {
                "object_type": "topic_slice",
                "group_id": "group_1001",
                "date_label": "2026-03-20",
                "topic_id": "20260320_0001",
                "core_text": "群里说开龙王局，就是紧急拉齐信息",
            },
        ]
    )
    service = SlangInterpretationService(
        llm_analysis_service=_StubLLMAnalysisService(),  # type: ignore[arg-type]
        embedding_backend=_StubEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=embedding_store,  # type: ignore[arg-type]
        slang_store=slang_store,
        enable_slang_learning=True,
        slang_retrieval_recent_days=7,
        slang_retrieval_limit=5,
        slang_min_context_items_for_inference=2,
    )
    context = _StubContext(
        responses=[
            (
                '{"slang_term":"龙王局","explanation":"群内表示临时高强度讨论局的说法",'
                '"usage_context":"突发任务时常用","confidence":0.82,"evidence_count":3,"no_info":false}'
            )
        ]
    )
    candidates = [
        SlangCandidate(
            term="龙王局",
            score=3.2,
            frequency=3,
            slice_coverage=2,
            evidence_count=2,
            source_slice_ids=["2026-03-22:20260322_0001"],
        )
    ]

    records, meta = _run(
        service.resolve_candidates(
            context=context,
            event=SimpleNamespace(unified_msg_origin="platform:group:1001"),
            analysis_provider_id="",
            group_id="group_1001",
            date_label="2026-03-22",
            candidates=candidates,
        )
    )

    assert embedding_store.calls
    assert len(records) == 1
    assert records[0].slang_term == "龙王局"
    assert records[0].explanation
    assert meta["inferred_count"] == 1
    persisted = slang_store.get(group_id="group_1001", slang_term="龙王局")
    assert persisted is not None


def test_slang_interpretation_service_degrades_when_context_insufficient(tmp_path: Path) -> None:
    slang_store = SlangStore(tmp_path / "slang")
    embedding_store = _StubEmbeddingStore(rows=[])
    service = SlangInterpretationService(
        llm_analysis_service=_StubLLMAnalysisService(),  # type: ignore[arg-type]
        embedding_backend=_StubEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=embedding_store,  # type: ignore[arg-type]
        slang_store=slang_store,
        enable_slang_learning=True,
        slang_min_context_items_for_inference=2,
    )
    context = _StubContext(responses=[])
    candidates = [
        SlangCandidate(
            term="龙王局",
            score=2.1,
            frequency=2,
            slice_coverage=2,
            evidence_count=2,
        )
    ]

    records, meta = _run(
        service.resolve_candidates(
            context=context,
            event=SimpleNamespace(unified_msg_origin="platform:group:1001"),
            analysis_provider_id="",
            group_id="group_1001",
            date_label="2026-03-22",
            candidates=candidates,
        )
    )

    assert records == []
    assert meta["insufficient_context_count"] == 1
    assert context.llm_calls == 0


def test_slang_interpretation_service_reuses_existing_explanations(tmp_path: Path) -> None:
    slang_store = SlangStore(tmp_path / "slang")
    existing = SlangExplanationRecord(
        group_id="group_1001",
        slang_term="龙王局",
        explanation="已有解释",
        usage_context="已有场景",
        confidence=0.8,
        evidence_count=5,
        created_at=int(datetime(2026, 3, 22, 10, 0, 0).timestamp()),
        updated_at=int(datetime(2026, 3, 22, 10, 0, 0).timestamp()),
    )
    slang_store.upsert(existing)
    service = SlangInterpretationService(
        llm_analysis_service=_StubLLMAnalysisService(),  # type: ignore[arg-type]
        embedding_backend=_StubEmbeddingBackend(),  # type: ignore[arg-type]
        embedding_store=_StubEmbeddingStore(rows=[]),  # type: ignore[arg-type]
        slang_store=slang_store,
        enable_slang_learning=True,
    )
    context = _StubContext(responses=[])
    candidates = [
        SlangCandidate(
            term="龙王局",
            score=2.1,
            frequency=2,
            slice_coverage=2,
            evidence_count=5,
        )
    ]

    records, meta = _run(
        service.resolve_candidates(
            context=context,
            event=SimpleNamespace(unified_msg_origin="platform:group:1001"),
            analysis_provider_id="",
            group_id="group_1001",
            date_label="2026-03-22",
            candidates=candidates,
        )
    )

    assert len(records) == 1
    assert records[0].explanation == "已有解释"
    assert meta["reused_count"] == 1
    assert meta["inferred_count"] == 0
    assert context.llm_calls == 0


def test_semantic_input_builder_injects_relevant_slang_contexts(tmp_path: Path) -> None:
    slang_store = SlangStore(tmp_path / "slang")
    slang_store.upsert(
        SlangExplanationRecord(
            group_id="group_1001",
            slang_term="龙王局",
            explanation="群内表示紧急高强度讨论局的说法",
            usage_context="出现突发任务时",
            confidence=0.87,
            evidence_count=4,
            created_at=int(datetime(2026, 3, 22, 10, 0, 0).timestamp()),
            updated_at=int(datetime(2026, 3, 22, 10, 0, 0).timestamp()),
        )
    )
    builder = SemanticInputBuilder(
        topic_segment_manager=_EmptyTopicManager(),  # type: ignore[arg-type]
        slang_store=slang_store,
        enable_slang_contexts=True,
    )
    material = _run(
        builder.build_for_full_window(
            group_id="group_1001",
            date_label="2026-03-22",
            time_window="2026-03-22 00:00 - 2026-03-22 23:59",
            mode="today",
            effective_messages=[
                MessageRecord(
                    "group_1001",
                    "u1",
                    "Alice",
                    "今天晚上要不要开个龙王局把预案再过一遍",
                    int(datetime(2026, 3, 22, 20, 0, 0).timestamp()),
                )
            ],
            max_messages_for_analysis=5,
        )
    )

    assert material.slang_context_count == 1
    assert any("slang_term=龙王局" in item for item in material.topic_slice_contexts)
    assert material.source.endswith("_plus_slang_contexts")
