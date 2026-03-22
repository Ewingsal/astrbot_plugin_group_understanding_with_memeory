from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from services.digest_service import ReportBuildMetrics
from services.group_origin_store import GroupOriginStore
from services.models import DigestReport, LLMAnalysisConfig, LLMSemanticResult, SchedulerConfig
from services.scheduler_service import SchedulerRuntimeOptions, ScheduledProactiveService


class _DigestStub:
    def __init__(
        self,
        reports_by_group: dict[str, DigestReport | None],
        sleep_by_group: dict[str, float] | None = None,
    ):
        self.reports_by_group = reports_by_group
        self.sleep_by_group = sleep_by_group or {}
        self.calls: list[dict] = []

    async def build_report_for_period_with_metrics(
        self,
        *,
        context,
        event,
        group_id,
        now,
        period,
        max_active_members,
        max_topics,
        analysis_config,
        mode=None,
        source="unknown",
    ):
        self.calls.append(
            {
                "group_id": group_id,
                "umo": getattr(event, "unified_msg_origin", ""),
                "period": period,
                "now": now,
                "max_active_members": max_active_members,
                "max_topics": max_topics,
                "analysis_provider_id": analysis_config.analysis_provider_id,
            }
        )
        sleep_seconds = float(self.sleep_by_group.get(group_id, 0.0))
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)
        return (
            self.reports_by_group.get(group_id),
            ReportBuildMetrics(load_messages_ms=1, aggregate_stats_ms=2, llm_analysis_ms=3),
        )


def _report(group_id: str, suggested_reply: str) -> DigestReport:
    return DigestReport(
        period="today",
        date_label="2026-03-22",
        time_window="2026-03-22 00:00 - 2026-03-22 18:00",
        group_id=group_id,
        total_messages=3,
        participant_count=2,
        active_members=[],
        llm_semantic=LLMSemanticResult(
            group_topics=["训练安排"],
            member_interests={"Alice": "关注训练"},
            overall_summary="群里讨论训练安排。",
            suggested_bot_reply=suggested_reply,
        ),
        stats_only=False,
        analysis_notice="语义分析模型：test_provider（configured）",
    )


def _run(coro):
    return asyncio.run(coro)


def _upsert(store: GroupOriginStore, **kwargs) -> None:
    _run(store.upsert_group_origin(**kwargs))


def _new_service(
    tmp_path: Path,
    *,
    reports_by_group: dict[str, DigestReport | None],
    sent: list[tuple[str, str]],
    sleep_by_group: dict[str, float] | None = None,
) -> tuple[ScheduledProactiveService, GroupOriginStore, _DigestStub]:
    store = GroupOriginStore(tmp_path / "group_origins.json")
    digest = _DigestStub(reports_by_group, sleep_by_group=sleep_by_group)

    async def _send(umo: str, text: str):
        sent.append((umo, text))

    service = ScheduledProactiveService(
        context=SimpleNamespace(),
        digest_service=digest,
        group_origin_store=store,
        send_func=_send,
    )
    return service, store, digest


def _configure_service(
    service: ScheduledProactiveService,
    *,
    enable: bool = False,
    whitelist_enabled: bool = False,
    whitelist: list[str] | None = None,
) -> None:
    service.start(
        scheduler_config=SchedulerConfig(
            enable_scheduled_proactive_message=enable,
            scheduled_send_hour=18,
            scheduled_send_minute=0,
            scheduled_mode="today_until_scheduled_time",
            store_group_origin=True,
            scheduled_group_whitelist_enabled=whitelist_enabled,
            scheduled_group_whitelist=list(whitelist or []),
            scheduled_send_timezone="Asia/Shanghai",
        ),
        analysis_config_builder=lambda: LLMAnalysisConfig(analysis_provider_id="test_provider"),
        runtime_options=SchedulerRuntimeOptions(
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
        ),
    )


def test_scheduler_does_not_send_without_group_origin(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, _store, digest = _new_service(tmp_path, reports_by_group={}, sent=sent)
    _configure_service(service, enable=False)

    result = _run(service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0)))

    assert result.total_records == 0
    assert result.sent_groups == []
    assert sent == []
    assert digest.calls == []


def test_scheduler_sends_per_group_without_cross_mix(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, store, digest = _new_service(
        tmp_path,
        reports_by_group={
            "group_a": _report("group_a", "A 群主动发言"),
            "group_b": _report("group_b", "B 群主动发言"),
        },
        sent=sent,
    )
    _configure_service(service, enable=False)

    _upsert(store, group_id="group_a", unified_msg_origin="umo_a", last_active_at=111)
    _upsert(store, group_id="group_b", unified_msg_origin="umo_b", last_active_at=222)

    result = _run(service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0)))

    assert result.generated_groups == ["group_a", "group_b"]
    assert result.sent_groups == ["group_a", "group_b"]
    assert sent == [("umo_a", "A 群主动发言"), ("umo_b", "B 群主动发言")]

    assert [call["group_id"] for call in digest.calls] == ["group_a", "group_b"]
    assert [call["umo"] for call in digest.calls] == ["umo_a", "umo_b"]


def test_scheduler_whitelist_only_sends_allowed_groups(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, store, _digest = _new_service(
        tmp_path,
        reports_by_group={
            "group_a": _report("group_a", "A 群主动发言"),
            "group_b": _report("group_b", "B 群主动发言"),
        },
        sent=sent,
    )
    _configure_service(service, enable=False, whitelist_enabled=True, whitelist=["group_a"])

    _upsert(store, group_id="group_a", unified_msg_origin="umo_a", last_active_at=111)
    _upsert(store, group_id="group_b", unified_msg_origin="umo_b", last_active_at=222)

    result = _run(service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0)))

    assert result.sent_groups == ["group_a"]
    assert result.skipped_whitelist == ["group_b"]
    assert sent == [("umo_a", "A 群主动发言")]


def test_scheduler_trigger_calls_send_logic(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, store, _digest = _new_service(
        tmp_path,
        reports_by_group={"group_a": _report("group_a", "A 群主动发言")},
        sent=sent,
    )
    _configure_service(service, enable=False)

    _upsert(store, group_id="group_a", unified_msg_origin="umo_a", last_active_at=111)

    _run(service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0)))

    assert sent == [("umo_a", "A 群主动发言")]


def test_private_chat_origin_not_stored_when_group_id_missing(tmp_path: Path) -> None:
    store = GroupOriginStore(tmp_path / "group_origins.json")

    _upsert(store, group_id="", unified_msg_origin="private:umo", last_active_at=123)

    assert store.list_group_records() == []


def test_scheduler_background_loop_runs_and_invokes_send(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, store, _digest = _new_service(
        tmp_path,
        reports_by_group={"group_a": _report("group_a", "A 群主动发言")},
        sent=sent,
    )
    _upsert(store, group_id="group_a", unified_msg_origin="umo_a", last_active_at=111)

    async def _case():
        service.start(
            scheduler_config=SchedulerConfig(
                enable_scheduled_proactive_message=True,
                scheduled_send_hour=18,
                scheduled_send_minute=0,
                scheduled_mode="today_until_scheduled_time",
                store_group_origin=True,
                scheduled_group_whitelist_enabled=False,
                scheduled_group_whitelist=[],
                scheduled_send_timezone="Asia/Shanghai",
            ),
            analysis_config_builder=lambda: LLMAnalysisConfig(analysis_provider_id="test_provider"),
            runtime_options=SchedulerRuntimeOptions(
                title_template="群聊兴趣日报（{date}）",
                max_active_members=5,
                max_topics=5,
            ),
        )

        # 直接触发一次，模拟“到点触发”后会调用发送逻辑。
        await service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0))
        await service.stop()

    _run(_case())

    assert sent == [("umo_a", "A 群主动发言")]


def test_scheduler_processes_groups_in_parallel(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    service, store, _digest = _new_service(
        tmp_path,
        reports_by_group={
            "group_a": _report("group_a", "A 群主动发言"),
            "group_b": _report("group_b", "B 群主动发言"),
        },
        sleep_by_group={"group_a": 0.12, "group_b": 0.12},
        sent=sent,
    )
    _configure_service(service, enable=False)

    _upsert(store, group_id="group_a", unified_msg_origin="umo_a", last_active_at=111)
    _upsert(store, group_id="group_b", unified_msg_origin="umo_b", last_active_at=222)

    started = perf_counter()
    _run(service.run_once_for_time(datetime(2026, 3, 22, 18, 0, 0)))
    elapsed = perf_counter() - started

    # 并行时总体耗时应明显小于串行的 ~0.24s。
    assert elapsed < 0.22
    assert len(sent) == 2
