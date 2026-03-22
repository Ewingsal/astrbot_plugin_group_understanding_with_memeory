from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from astrbot.api import logger

from .digest_service import GroupDigestService
from .group_origin_store import GroupOriginRecord, GroupOriginStore
from .models import LLMAnalysisConfig, SchedulerConfig


@dataclass(frozen=True)
class SchedulerRuntimeOptions:
    title_template: str
    max_active_members: int
    max_topics: int


@dataclass(frozen=True)
class SchedulerRunResult:
    trigger_time: datetime
    total_records: int
    traversed_groups: list[str]
    skipped_missing_origin: list[str]
    skipped_whitelist: list[str]
    skipped_no_messages: list[str]
    skipped_no_suggestion: list[str]
    generated_groups: list[str]
    sent_groups: list[str]
    failed_groups: dict[str, str]
    processed_groups: int
    successful_groups: int
    total_scheduler_ms: int


@dataclass(frozen=True)
class _GroupProcessResult:
    group_id: str
    status: str
    reason: str = ""
    load_messages_ms: int = 0
    aggregate_stats_ms: int = 0
    llm_analysis_ms: int = 0
    send_message_ms: int = 0
    total_group_ms: int = 0
    provider_notice: str = ""


class ScheduledProactiveService:
    """每日固定时间主动发言调度服务（按群隔离）。"""

    def __init__(
        self,
        *,
        context: Any,
        digest_service: GroupDigestService,
        group_origin_store: GroupOriginStore,
        send_func: Callable[[str, str], Awaitable[None]] | None = None,
        now_func: Callable[[tzinfo], datetime] | None = None,
    ) -> None:
        self.context = context
        self.digest_service = digest_service
        self.group_origin_store = group_origin_store
        self._send_func = send_func or self._default_send_message
        self._now_func = now_func or (lambda tz: datetime.now(tz))

        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._is_running = False

        self._scheduler_config = SchedulerConfig()
        self._analysis_config_builder: Callable[[], LLMAnalysisConfig] | None = None
        self._runtime_options = SchedulerRuntimeOptions(
            title_template="群聊兴趣日报（{date}）",
            max_active_members=5,
            max_topics=5,
        )
        self._timezone = datetime.now().astimezone().tzinfo or timezone.utc

    def start(
        self,
        *,
        scheduler_config: SchedulerConfig,
        analysis_config_builder: Callable[[], LLMAnalysisConfig],
        runtime_options: SchedulerRuntimeOptions,
    ) -> None:
        self._scheduler_config = scheduler_config
        self._analysis_config_builder = analysis_config_builder
        self._runtime_options = runtime_options
        self._timezone = self._resolve_timezone(scheduler_config.scheduled_send_timezone)

        if not scheduler_config.enable_scheduled_proactive_message:
            logger.info("[group_digest.scheduler] disabled by config")
            return

        if self._task and not self._task.done():
            logger.warning("[group_digest.scheduler] already running, skip duplicate start")
            return

        self._stop_event = None
        self._is_running = True
        try:
            self._task = asyncio.create_task(self._run_loop())
        except RuntimeError:
            self._is_running = False
            logger.exception("[group_digest.scheduler] failed to start: no running event loop")
            return

        logger.info(
            "[group_digest.scheduler] started. time=%02d:%02d mode=%s timezone=%s whitelist_enabled=%s",
            scheduler_config.scheduled_send_hour,
            scheduler_config.scheduled_send_minute,
            scheduler_config.scheduled_mode,
            scheduler_config.scheduled_send_timezone,
            scheduler_config.scheduled_group_whitelist_enabled,
        )

    async def stop(self) -> None:
        self._is_running = False
        if self._stop_event is not None:
            self._stop_event.set()

        if not self._task:
            return

        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("[group_digest.scheduler] stop failed")

        self._task = None
        logger.info("[group_digest.scheduler] stopped")

    async def run_once_for_time(self, trigger_time: datetime) -> SchedulerRunResult:
        """执行一次调度。用于调度循环，也可用于联调与单元测试。"""
        scheduler_start = perf_counter()
        config = self._scheduler_config
        if config.scheduled_mode != "today_until_scheduled_time":
            logger.warning(
                "[group_digest.scheduler] unsupported scheduled_mode=%s. skip run",
                config.scheduled_mode,
            )
            return SchedulerRunResult(
                trigger_time=trigger_time,
                total_records=0,
                traversed_groups=[],
                skipped_missing_origin=[],
                skipped_whitelist=[],
                skipped_no_messages=[],
                skipped_no_suggestion=[],
                generated_groups=[],
                sent_groups=[],
                failed_groups={},
                processed_groups=0,
                successful_groups=0,
                total_scheduler_ms=int((perf_counter() - scheduler_start) * 1000),
            )

        records = self.group_origin_store.list_group_records()
        traversed_groups = [record.group_id for record in records]

        logger.info(
            "[group_digest.scheduler] trigger at %s. total_groups=%d groups=%s",
            trigger_time.isoformat(timespec="seconds"),
            len(records),
            traversed_groups,
        )

        whitelist = set(config.scheduled_group_whitelist)
        analysis_config = self._analysis_config_builder() if self._analysis_config_builder else LLMAnalysisConfig()

        tasks = [
            self._process_single_group(
                record=record,
                trigger_time=trigger_time,
                whitelist=whitelist,
                whitelist_enabled=config.scheduled_group_whitelist_enabled,
                analysis_config=analysis_config,
            )
            for record in records
        ]
        group_results = await asyncio.gather(*tasks) if tasks else []

        skipped_missing_origin: list[str] = []
        skipped_whitelist: list[str] = []
        skipped_no_messages: list[str] = []
        skipped_no_suggestion: list[str] = []
        generated_groups: list[str] = []
        sent_groups: list[str] = []
        failed_groups: dict[str, str] = {}

        for item in group_results:
            if item.status == "skipped_missing_origin":
                skipped_missing_origin.append(item.group_id)
            elif item.status == "skipped_whitelist":
                skipped_whitelist.append(item.group_id)
            elif item.status == "skipped_no_messages":
                skipped_no_messages.append(item.group_id)
            elif item.status == "skipped_no_suggestion":
                skipped_no_suggestion.append(item.group_id)
                generated_groups.append(item.group_id)
            elif item.status == "failed":
                generated_groups.append(item.group_id)
                failed_groups[item.group_id] = item.reason
            elif item.status == "sent":
                generated_groups.append(item.group_id)
                sent_groups.append(item.group_id)

        processed_groups = len(
            [
                item
                for item in group_results
                if item.status not in {"skipped_missing_origin", "skipped_whitelist"}
            ]
        )
        successful_groups = len(sent_groups)
        total_scheduler_ms = int((perf_counter() - scheduler_start) * 1000)

        logger.info(
            "[group_digest.scheduler.summary] trigger=%s total_scheduler_ms=%d total_groups=%d processed_groups=%d successful_groups=%d",
            trigger_time.isoformat(timespec="seconds"),
            total_scheduler_ms,
            len(records),
            processed_groups,
            successful_groups,
        )

        return SchedulerRunResult(
            trigger_time=trigger_time,
            total_records=len(records),
            traversed_groups=traversed_groups,
            skipped_missing_origin=skipped_missing_origin,
            skipped_whitelist=skipped_whitelist,
            skipped_no_messages=skipped_no_messages,
            skipped_no_suggestion=skipped_no_suggestion,
            generated_groups=generated_groups,
            sent_groups=sent_groups,
            failed_groups=failed_groups,
            processed_groups=processed_groups,
            successful_groups=successful_groups,
            total_scheduler_ms=total_scheduler_ms,
        )

    async def _process_single_group(
        self,
        *,
        record: GroupOriginRecord,
        trigger_time: datetime,
        whitelist: set[str],
        whitelist_enabled: bool,
        analysis_config: LLMAnalysisConfig,
    ) -> _GroupProcessResult:
        group_start = perf_counter()

        if not self._is_valid_record(record):
            result = _GroupProcessResult(
                group_id=record.group_id,
                status="skipped_missing_origin",
                reason="missing_unified_msg_origin",
                total_group_ms=int((perf_counter() - group_start) * 1000),
            )
            self._log_group_timing(result)
            return result

        if whitelist_enabled and record.group_id not in whitelist:
            result = _GroupProcessResult(
                group_id=record.group_id,
                status="skipped_whitelist",
                reason="not_in_whitelist",
                total_group_ms=int((perf_counter() - group_start) * 1000),
            )
            self._log_group_timing(result)
            return result

        report, metrics = await self.digest_service.build_report_for_period_with_metrics(
            context=self.context,
            event=SimpleNamespace(unified_msg_origin=record.unified_msg_origin),
            group_id=record.group_id,
            now=trigger_time,
            period="today",
            max_active_members=self._runtime_options.max_active_members,
            max_topics=self._runtime_options.max_topics,
            analysis_config=analysis_config,
            mode="scheduled",
            source="scheduler",
        )

        if report is None:
            result = _GroupProcessResult(
                group_id=record.group_id,
                status="skipped_no_messages",
                reason="no_messages_in_window",
                load_messages_ms=metrics.load_messages_ms,
                aggregate_stats_ms=metrics.aggregate_stats_ms,
                llm_analysis_ms=metrics.llm_analysis_ms,
                total_group_ms=int((perf_counter() - group_start) * 1000),
            )
            self._log_group_timing(result)
            return result

        suggested_reply = ""
        if report.llm_semantic and report.llm_semantic.suggested_bot_reply:
            suggested_reply = report.llm_semantic.suggested_bot_reply.strip()

        if not suggested_reply:
            result = _GroupProcessResult(
                group_id=record.group_id,
                status="skipped_no_suggestion",
                reason="no_suggested_bot_reply",
                load_messages_ms=metrics.load_messages_ms,
                aggregate_stats_ms=metrics.aggregate_stats_ms,
                llm_analysis_ms=metrics.llm_analysis_ms,
                total_group_ms=int((perf_counter() - group_start) * 1000),
                provider_notice=report.analysis_notice,
            )
            self._log_group_timing(result)
            return result

        send_start = perf_counter()
        try:
            await self._send_func(record.unified_msg_origin, suggested_reply)
        except Exception as exc:
            result = _GroupProcessResult(
                group_id=record.group_id,
                status="failed",
                reason=str(exc),
                load_messages_ms=metrics.load_messages_ms,
                aggregate_stats_ms=metrics.aggregate_stats_ms,
                llm_analysis_ms=metrics.llm_analysis_ms,
                send_message_ms=int((perf_counter() - send_start) * 1000),
                total_group_ms=int((perf_counter() - group_start) * 1000),
                provider_notice=report.analysis_notice,
            )
            self._log_group_timing(result)
            return result

        result = _GroupProcessResult(
            group_id=record.group_id,
            status="sent",
            load_messages_ms=metrics.load_messages_ms,
            aggregate_stats_ms=metrics.aggregate_stats_ms,
            llm_analysis_ms=metrics.llm_analysis_ms,
            send_message_ms=int((perf_counter() - send_start) * 1000),
            total_group_ms=int((perf_counter() - group_start) * 1000),
            provider_notice=report.analysis_notice,
        )
        self._log_group_timing(result)
        return result

    def _log_group_timing(self, result: _GroupProcessResult) -> None:
        logger.info(
            "[group_digest.scheduler.group_timing] group=%s status=%s reason=%s load_messages_ms=%d aggregate_stats_ms=%d llm_analysis_ms=%d send_message_ms=%d total_group_ms=%d provider=%s",
            result.group_id,
            result.status,
            result.reason or "-",
            result.load_messages_ms,
            result.aggregate_stats_ms,
            result.llm_analysis_ms,
            result.send_message_ms,
            result.total_group_ms,
            result.provider_notice or "-",
        )

    async def _run_loop(self) -> None:
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        while self._is_running:
            try:
                now_local = self._now_func(self._timezone)
                next_run = self._compute_next_run(now_local)
                wait_seconds = max((next_run - now_local).total_seconds(), 1.0)

                logger.info(
                    "[group_digest.scheduler] next_run=%s wait_seconds=%.2f",
                    next_run.isoformat(timespec="seconds"),
                    wait_seconds,
                )

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                    break
                except asyncio.TimeoutError:
                    pass

                await self.run_once_for_time(trigger_time=next_run)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[group_digest.scheduler] loop_error")
                await asyncio.sleep(5)

    def _compute_next_run(self, now_local: datetime) -> datetime:
        config = self._scheduler_config
        target = now_local.replace(
            hour=config.scheduled_send_hour,
            minute=config.scheduled_send_minute,
            second=0,
            microsecond=0,
        )
        if target <= now_local:
            target = target + timedelta(days=1)
        return target

    def _resolve_timezone(self, tz_name: str):
        tz_name = (tz_name or "").strip()
        if not tz_name:
            return datetime.now().astimezone().tzinfo or timezone.utc

        if ZoneInfo is None:  # pragma: no cover - Python < 3.9 fallback
            logger.warning(
                "[group_digest.scheduler] zoneinfo unavailable, fallback to local timezone. configured=%s",
                tz_name,
            )
            return datetime.now().astimezone().tzinfo or timezone.utc

        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning(
                "[group_digest.scheduler] invalid timezone=%s, fallback to local timezone",
                tz_name,
            )
            return datetime.now().astimezone().tzinfo or timezone.utc

    def _is_valid_record(self, record: GroupOriginRecord) -> bool:
        return bool(record.group_id and record.unified_msg_origin)

    async def _default_send_message(self, unified_msg_origin: str, text: str) -> None:
        sender = getattr(self.context, "send_message", None)
        if not callable(sender):
            raise RuntimeError("当前 AstrBot 上下文未提供 send_message 接口")

        # TODO: 若未来 AstrBot send_message 签名变化，可在此适配层统一处理。
        try:
            from astrbot.api.event import MessageChain

            chain = MessageChain().message(text)
        except Exception:
            chain = text

        await sender(unified_msg_origin, chain)
