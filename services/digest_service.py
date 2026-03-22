from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from astrbot.api import logger

from .interaction_service import InteractionService
from .llm_analysis_service import LLMAnalysisService
from .message_filters import effective_message_stats, filter_effective_messages
from .models import DigestReport, LLMAnalysisConfig, LLMSemanticResult, MemberDigest, MessageRecord
from .report_cache_store import ReportCacheRecord, ReportCacheStore
from .storage import JsonMessageStorage

PeriodType = Literal["yesterday", "today"]
ReportMode = Literal["yesterday", "today", "scheduled"]


@dataclass(frozen=True)
class ReportWindow:
    period: PeriodType
    date_label: str
    start_ts: int
    end_ts: int
    time_window: str


@dataclass(frozen=True)
class ReportBuildMetrics:
    load_messages_ms: int
    aggregate_stats_ms: int
    llm_analysis_ms: int


class GroupDigestService:
    def __init__(
        self,
        storage: JsonMessageStorage,
        llm_analysis_service: LLMAnalysisService,
        interaction_service: InteractionService,
        template_path: Path,
        report_cache_store: ReportCacheStore | None = None,
        cache_version: int = 1,
    ):
        self.storage = storage
        self.llm_analysis_service = llm_analysis_service
        self.interaction_service = interaction_service
        self.template_path = template_path
        self.report_cache_store = report_cache_store
        self.cache_version = int(cache_version)

    async def generate_digest_text_for_period(
        self,
        *,
        context: Any,
        event: Any,
        group_id: str,
        now: datetime,
        period: PeriodType,
        title_template: str,
        max_active_members: int,
        max_topics: int,
        analysis_config: LLMAnalysisConfig,
        source: str,
    ) -> str:
        mode: ReportMode = "scheduled" if source == "scheduler" else period
        report = await self.build_report_for_period(
            context=context,
            event=event,
            group_id=group_id,
            now=now,
            period=period,
            max_active_members=max_active_members,
            max_topics=max_topics,
            analysis_config=analysis_config,
            mode=mode,
            source=source,
        )

        window = self._resolve_report_window(now=now, period=period)
        if report is None:
            return self._render_no_messages(
                group_id=group_id,
                title_template=title_template,
                window=window,
            )

        if report.analysis_notice and report.analysis_notice.startswith("[ERROR]"):
            return self._render_analysis_error(
                report=report,
                title_template=title_template,
            )

        return self.render_text(report=report, title_template=title_template)

    async def build_report_for_period(
        self,
        *,
        context: Any,
        event: Any,
        group_id: str,
        now: datetime,
        period: PeriodType,
        max_active_members: int = 5,
        max_topics: int = 5,
        analysis_config: LLMAnalysisConfig | None = None,
        mode: ReportMode | None = None,
        source: str = "unknown",
    ) -> DigestReport | None:
        report, _metrics = await self.build_report_for_period_with_metrics(
            context=context,
            event=event,
            group_id=group_id,
            now=now,
            period=period,
            max_active_members=max_active_members,
            max_topics=max_topics,
            analysis_config=analysis_config,
            mode=mode,
            source=source,
        )
        return report

    async def build_report_for_period_with_metrics(
        self,
        *,
        context: Any,
        event: Any,
        group_id: str,
        now: datetime,
        period: PeriodType,
        max_active_members: int = 5,
        max_topics: int = 5,
        analysis_config: LLMAnalysisConfig | None = None,
        mode: ReportMode | None = None,
        source: str = "unknown",
    ) -> tuple[DigestReport | None, ReportBuildMetrics]:
        window = self._resolve_report_window(now=now, period=period)
        config = analysis_config or LLMAnalysisConfig()
        cache_mode: ReportMode = mode or period
        prompt_signature = self._build_prompt_signature(config=config, max_topics=max_topics)

        load_start = perf_counter()
        raw_messages = self.storage.load_messages(
            group_id=group_id,
            start_ts=window.start_ts,
            end_ts=window.end_ts,
        )
        load_messages_ms = int((perf_counter() - load_start) * 1000)
        effective_messages, excluded_reasons = filter_effective_messages(raw_messages)
        message_count, last_message_ts = effective_message_stats(effective_messages)
        excluded_total = sum(excluded_reasons.values())
        if excluded_total:
            logger.info(
                "[group_digest.filter] group_id=%s date=%s mode=%s raw_message_count=%d effective_message_count=%d excluded_total=%d excluded_plugin_command=%d excluded_plugin_output_prefix=%d excluded_plugin_sender_id=%d",
                group_id,
                window.date_label,
                cache_mode,
                len(raw_messages),
                message_count,
                excluded_total,
                excluded_reasons.get("plugin_command", 0),
                excluded_reasons.get("plugin_output_prefix", 0),
                excluded_reasons.get("plugin_sender_id", 0),
            )

        if message_count <= 0:
            return (
                None,
                ReportBuildMetrics(
                    load_messages_ms=load_messages_ms,
                    aggregate_stats_ms=0,
                    llm_analysis_ms=0,
                ),
            )

        expected_provider_id, expected_provider_source, expected_provider_err = await self._resolve_expected_provider_id(
            context=context,
            event=event,
            config=config,
        )

        if self.report_cache_store is not None:
            cache_record = self.report_cache_store.get_record(
                group_id=group_id,
                date=window.date_label,
                mode=cache_mode,
            )
            cache_miss_reason = "no_cache_entry"
            if cache_record is not None:
                cache_miss_reason = self._validate_cache_record(
                    cache_record=cache_record,
                    current_message_count=message_count,
                    current_last_message_ts=last_message_ts,
                    expected_provider_id=expected_provider_id,
                    expected_provider_err=expected_provider_err,
                    max_messages_for_analysis=config.max_messages_for_analysis,
                    prompt_signature=prompt_signature,
                    window_start=window.start_ts,
                )
                if not cache_miss_reason:
                    cached_report = self._report_from_payload(cache_record.report)
                    if cached_report is not None:
                        logger.info(
                            "[group_digest.cache] cache_hit group_id=%s date=%s mode=%s reason=no_new_messages_and_config_stable effective_message_count=%s effective_last_message_ts=%s",
                            group_id,
                            window.date_label,
                            cache_mode,
                            message_count,
                            last_message_ts,
                        )
                        return (
                            cached_report,
                            ReportBuildMetrics(
                                load_messages_ms=load_messages_ms,
                                aggregate_stats_ms=0,
                                llm_analysis_ms=0,
                            ),
                        )
                    cache_miss_reason = "cached_report_invalid"

            if cache_miss_reason == "no_cache_entry":
                logger.info(
                    "[group_digest.cache] cache_miss group_id=%s date=%s mode=%s reason=%s effective_message_count=%s effective_last_message_ts=%s",
                    group_id,
                    window.date_label,
                    cache_mode,
                    cache_miss_reason,
                    message_count,
                    last_message_ts,
                )
            elif cache_miss_reason in {"new_messages_detected", "provider_changed", "max_messages_changed", "prompt_signature_changed"}:
                logger.info(
                    "[group_digest.cache] cache_refresh group_id=%s date=%s mode=%s reason=%s effective_message_count=%s effective_last_message_ts=%s",
                    group_id,
                    window.date_label,
                    cache_mode,
                    cache_miss_reason,
                    message_count,
                    last_message_ts,
                )
            else:
                logger.info(
                    "[group_digest.cache] cache_miss group_id=%s date=%s mode=%s reason=%s effective_message_count=%s effective_last_message_ts=%s",
                    group_id,
                    window.date_label,
                    cache_mode,
                    cache_miss_reason,
                    message_count,
                    last_message_ts,
                )

        report, metrics, rebuilt_count, rebuilt_last_ts = await self._build_report_without_cache(
            context=context,
            event=event,
            group_id=group_id,
            window=window,
            period=period,
            max_active_members=max_active_members,
            max_topics=max_topics,
            analysis_config=config,
            expected_provider_id=expected_provider_id,
            expected_provider_source=expected_provider_source,
            effective_messages=effective_messages,
            load_messages_ms=load_messages_ms,
        )

        if report is not None and self.report_cache_store is not None:
            cache_provider_id = report.analysis_provider_id or expected_provider_id
            record = ReportCacheRecord(
                group_id=group_id,
                date=window.date_label,
                mode=cache_mode,
                window_start=window.start_ts,
                window_end=window.end_ts,
                generated_at=datetime.now().isoformat(timespec="seconds"),
                last_message_timestamp=rebuilt_last_ts,
                message_count=rebuilt_count,
                provider_id=cache_provider_id,
                analysis_provider_notice=report.analysis_notice,
                max_messages_for_analysis=config.max_messages_for_analysis,
                prompt_signature=prompt_signature,
                cache_version=self.cache_version,
                source=source,
                report=self._report_to_payload(report),
            )
            await self.report_cache_store.upsert_record(record)
            logger.info(
                "[group_digest.cache] cache_write group_id=%s date=%s mode=%s source=%s message_count=%s",
                group_id,
                window.date_label,
                cache_mode,
                source,
                rebuilt_count,
            )

        return report, metrics

    async def _build_report_without_cache(
        self,
        *,
        context: Any,
        event: Any,
        group_id: str,
        window: ReportWindow,
        period: PeriodType,
        max_active_members: int,
        max_topics: int,
        analysis_config: LLMAnalysisConfig,
        expected_provider_id: str,
        expected_provider_source: str,
        effective_messages: list[MessageRecord],
        load_messages_ms: int,
    ) -> tuple[DigestReport | None, ReportBuildMetrics, int, int]:
        if not effective_messages:
            return (
                None,
                ReportBuildMetrics(
                    load_messages_ms=load_messages_ms,
                    aggregate_stats_ms=0,
                    llm_analysis_ms=0,
                ),
                0,
                0,
            )

        aggregate_start = perf_counter()
        report = self._build_stats_report(
            period=period,
            group_id=group_id,
            window=window,
            messages=effective_messages,
            max_active_members=max_active_members,
        )
        aggregate_stats_ms = int((perf_counter() - aggregate_start) * 1000)

        llm_start = perf_counter()
        outcome = await self.llm_analysis_service.analyze(
            context=context,
            event=event,
            config=analysis_config,
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            messages=effective_messages,
            active_members=report.active_members,
            max_topics=max_topics,
            resolved_provider_id=expected_provider_id if expected_provider_id else None,
            resolved_provider_source=expected_provider_source,
        )
        llm_analysis_ms = int((perf_counter() - llm_start) * 1000)

        if outcome.semantic is not None:
            outcome.semantic.suggested_bot_reply = self.interaction_service.finalize_suggested_reply(
                outcome.semantic.suggested_bot_reply
            )
            report.llm_semantic = outcome.semantic
            report.stats_only = False
            report.analysis_notice = outcome.notice
            report.analysis_provider_id = outcome.provider_id
        elif outcome.error:
            report.analysis_provider_id = outcome.provider_id
            if analysis_config.fallback_to_stats_only:
                report.stats_only = True
                report.analysis_notice = f"语义分析失败，已降级为仅统计：{outcome.error}"
            else:
                report.stats_only = True
                report.analysis_notice = f"[ERROR] 语义分析失败：{outcome.error}"
        else:
            report.stats_only = True
            report.analysis_notice = outcome.notice or "语义分析已关闭，当前仅展示统计结果。"
            report.analysis_provider_id = outcome.provider_id

        return (
            report,
            ReportBuildMetrics(
                load_messages_ms=load_messages_ms,
                aggregate_stats_ms=aggregate_stats_ms,
                llm_analysis_ms=llm_analysis_ms,
            ),
            len(effective_messages),
            max((row.timestamp for row in effective_messages), default=0),
        )

    async def _resolve_expected_provider_id(
        self,
        *,
        context: Any,
        event: Any,
        config: LLMAnalysisConfig,
    ) -> tuple[str, str, str]:
        if not config.use_llm_topic_analysis:
            return "", "", ""

        provider_id, source, error = await self.llm_analysis_service.resolve_provider_id(
            context=context,
            event=event,
            configured_provider_id=config.analysis_provider_id,
        )
        if provider_id:
            return provider_id, source, ""
        return "", "", error or "provider_unavailable"

    def _validate_cache_record(
        self,
        *,
        cache_record: ReportCacheRecord,
        current_message_count: int,
        current_last_message_ts: int,
        expected_provider_id: str,
        expected_provider_err: str,
        max_messages_for_analysis: int,
        prompt_signature: str,
        window_start: int,
    ) -> str:
        if cache_record.cache_version != self.cache_version:
            return "cache_version_changed"
        if cache_record.window_start != window_start:
            return "window_changed"
        if cache_record.max_messages_for_analysis != max_messages_for_analysis:
            return "max_messages_changed"
        if cache_record.prompt_signature != prompt_signature:
            return "prompt_signature_changed"
        if expected_provider_err:
            return "provider_validation_failed"
        if expected_provider_id and cache_record.provider_id != expected_provider_id:
            return "provider_changed"
        if cache_record.message_count != current_message_count:
            return "new_messages_detected"
        if cache_record.last_message_timestamp != current_last_message_ts:
            return "new_messages_detected"
        return ""

    def _build_prompt_signature(self, *, config: LLMAnalysisConfig, max_topics: int) -> str:
        payload = {
            "analysis_prompt_template": config.analysis_prompt_template.strip(),
            "interaction_prompt_template": config.interaction_prompt_template.strip(),
            "use_llm_topic_analysis": config.use_llm_topic_analysis,
            "max_topics": max_topics,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _report_to_payload(self, report: DigestReport) -> dict[str, Any]:
        return {
            "period": report.period,
            "date_label": report.date_label,
            "time_window": report.time_window,
            "group_id": report.group_id,
            "total_messages": report.total_messages,
            "participant_count": report.participant_count,
            "active_members": [
                {
                    "sender_id": member.sender_id,
                    "sender_name": member.sender_name,
                    "message_count": member.message_count,
                }
                for member in report.active_members
            ],
            "llm_semantic": {
                "group_topics": report.llm_semantic.group_topics,
                "member_interests": report.llm_semantic.member_interests,
                "overall_summary": report.llm_semantic.overall_summary,
                "suggested_bot_reply": report.llm_semantic.suggested_bot_reply,
            }
            if report.llm_semantic
            else None,
            "stats_only": report.stats_only,
            "analysis_notice": report.analysis_notice,
            "analysis_provider_id": report.analysis_provider_id,
        }

    def _report_from_payload(self, payload: dict[str, Any]) -> DigestReport | None:
        if not isinstance(payload, dict):
            logger.warning(
                "[group_digest.cache] cached_report_invalid_payload expected=dict got=%s",
                type(payload).__name__,
            )
            return None

        try:
            active_members_raw = payload.get("active_members", [])
            active_members: list[MemberDigest] = []
            if isinstance(active_members_raw, list):
                for item in active_members_raw:
                    if not isinstance(item, dict):
                        continue
                    active_members.append(
                        MemberDigest(
                            sender_id=str(item.get("sender_id", "")),
                            sender_name=str(item.get("sender_name", "")),
                            message_count=self._safe_int(item.get("message_count", 0), field="active_member.message_count"),
                        )
                    )

            report = DigestReport(
                period=str(payload.get("period", "")),
                date_label=str(payload.get("date_label", "")),
                time_window=str(payload.get("time_window", "")),
                group_id=str(payload.get("group_id", "")),
                total_messages=self._safe_int(payload.get("total_messages", 0), field="report.total_messages"),
                participant_count=self._safe_int(payload.get("participant_count", 0), field="report.participant_count"),
                active_members=active_members,
                stats_only=bool(payload.get("stats_only", False)),
                analysis_notice=str(payload.get("analysis_notice", "")),
                analysis_provider_id=str(payload.get("analysis_provider_id", "")),
            )

            semantic_raw = payload.get("llm_semantic", None)
            if isinstance(semantic_raw, dict):
                report.llm_semantic = LLMSemanticResult(
                    group_topics=[
                        str(item).strip()
                        for item in semantic_raw.get("group_topics", [])
                        if str(item).strip()
                    ],
                    member_interests={
                        str(name).strip(): str(summary).strip()
                        for name, summary in semantic_raw.get("member_interests", {}).items()
                            if str(name).strip() and str(summary).strip()
                    },
                    overall_summary=str(semantic_raw.get("overall_summary", "")).strip(),
                    suggested_bot_reply=str(semantic_raw.get("suggested_bot_reply", "")).strip(),
                )
            return report
        except Exception as exc:
            logger.warning(
                "[group_digest.cache] cached_report_parse_failed error=%s payload_keys=%s",
                exc,
                sorted(payload.keys()),
            )
            return None

    def _safe_int(self, value: object, *, field: str, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.cache] invalid_int field=%s value=%r fallback=%d",
                field,
                value,
                default,
            )
            return default

    def generate_today_debug_text(
        self,
        group_id: str,
        now: datetime,
        max_active_members: int,
    ) -> str:
        """调试输出：专注今日归档计数。"""
        window = self._resolve_report_window(now=now, period="today")
        messages = self.storage.load_messages(
            group_id=group_id,
            start_ts=window.start_ts,
            end_ts=window.end_ts,
        )
        messages, _excluded = filter_effective_messages(messages)

        if not messages:
            return (
                "[调试] 今日消息统计\n"
                f"统计日期：{window.date_label}\n"
                f"统计范围：{window.time_window}\n"
                f"群组：{group_id}\n"
                "消息总数：0\n"
                "参与成员：0\n"
                "结果：今天暂无消息。"
            )

        counts: dict[str, tuple[str, int]] = {}
        for row in messages:
            display_name, current = counts.get(row.sender_id, (row.sender_name, 0))
            counts[row.sender_id] = (display_name, current + 1)

        top_rows = sorted(counts.values(), key=lambda item: (-item[1], item[0]))[:max_active_members]
        top_block = "\n".join(f"{idx}. {name}（{cnt} 条）" for idx, (name, cnt) in enumerate(top_rows, start=1))

        return (
            "[调试] 今日消息统计\n"
            f"统计日期：{window.date_label}\n"
            f"统计范围：{window.time_window}\n"
            f"群组：{group_id}\n"
            f"消息总数：{len(messages)}\n"
            f"参与成员：{len(counts)}\n"
            "活跃成员 Top：\n"
            f"{top_block}"
        )

    def render_text(self, report: DigestReport, title_template: str) -> str:
        title = title_template.format(date=report.date_label)
        active_members_block = self._render_active_members(report.active_members)

        if report.stats_only or report.llm_semantic is None:
            return self._render_stats_only_text(
                title=title,
                report=report,
                active_members_block=active_members_block,
            )

        semantic = report.llm_semantic
        group_topics_block = self._render_topics(semantic.group_topics)
        member_interest_block = self._render_member_interests(semantic.member_interests)

        template = self._load_template()
        return template.format(
            title=title,
            date_label=report.date_label,
            time_window=report.time_window,
            group_id=report.group_id,
            total_messages=report.total_messages,
            participant_count=report.participant_count,
            active_members_block=active_members_block,
            overall_summary=semantic.overall_summary,
            group_topics_block=group_topics_block,
            member_interest_block=member_interest_block,
            suggested_bot_reply=semantic.suggested_bot_reply,
            analysis_notice=report.analysis_notice or "语义分析已启用。",
        )

    def _render_stats_only_text(self, title: str, report: DigestReport, active_members_block: str) -> str:
        notice = report.analysis_notice or "当前仅输出统计结果。"
        return (
            f"{title}\n"
            f"统计日期：{report.date_label}\n"
            f"统计范围：{report.time_window}\n"
            f"群组：{report.group_id}\n\n"
            "【统计总览】\n"
            f"- 总发言数：{report.total_messages}\n"
            f"- 参与成员数：{report.participant_count}\n\n"
            "【活跃成员排行】\n"
            f"{active_members_block}\n\n"
            "【语义分析状态】\n"
            f"{notice}"
        )

    def _render_no_messages(self, group_id: str, title_template: str, window: ReportWindow) -> str:
        title = title_template.format(date=window.date_label)

        if window.period == "today":
            no_data_line = "今日暂无消息。"
            guide_line = "建议先让群内产生聊天后再执行 `/group_digest_today`。"
        else:
            no_data_line = "暂无昨日消息。"
            guide_line = "建议先让群内产生聊天后再执行 `/group_digest`。"

        return (
            f"{title}\n"
            f"统计日期：{window.date_label}\n"
            f"统计范围：{window.time_window}\n"
            f"群组：{group_id}\n\n"
            f"{no_data_line}\n"
            f"{guide_line}"
        )

    def _render_analysis_error(self, report: DigestReport, title_template: str) -> str:
        title = title_template.format(date=report.date_label)
        reason = report.analysis_notice.replace("[ERROR] ", "").strip()
        return (
            f"{title}\n"
            f"统计日期：{report.date_label}\n"
            f"统计范围：{report.time_window}\n"
            f"群组：{report.group_id}\n\n"
            "语义分析不可用，且未启用统计降级。\n"
            f"原因：{reason}\n"
            "请检查模型配置后重试。"
        )

    def _build_stats_report(
        self,
        *,
        period: PeriodType,
        group_id: str,
        window: ReportWindow,
        messages: list[MessageRecord],
        max_active_members: int,
    ) -> DigestReport:
        members_map: dict[str, list[MessageRecord]] = {}
        for row in messages:
            members_map.setdefault(row.sender_id, []).append(row)

        active_members: list[MemberDigest] = []
        for sender_id, rows in members_map.items():
            active_members.append(
                MemberDigest(
                    sender_id=sender_id,
                    sender_name=rows[0].sender_name,
                    message_count=len(rows),
                )
            )

        active_members.sort(key=lambda item: (-item.message_count, item.sender_name))
        active_members = active_members[:max_active_members]

        return DigestReport(
            period=period,
            date_label=window.date_label,
            time_window=window.time_window,
            group_id=group_id,
            total_messages=len(messages),
            participant_count=len(members_map),
            active_members=active_members,
        )

    def _render_active_members(self, members: list[MemberDigest]) -> str:
        if not members:
            return "1. 暂无数据"
        lines = [f"{idx}. {m.sender_name}（{m.message_count} 条）" for idx, m in enumerate(members, start=1)]
        return "\n".join(lines)

    def _render_topics(self, topics: list[str]) -> str:
        if not topics:
            return "1. 模型未返回热门话题"
        return "\n".join(f"{idx}. {topic}" for idx, topic in enumerate(topics, start=1))

    def _render_member_interests(self, interests: dict[str, str]) -> str:
        if not interests:
            return "- 模型未返回成员兴趣摘要"
        lines: list[str] = []
        for name, summary in interests.items():
            lines.append(f"- {name}：{summary}")
        return "\n".join(lines)

    def _resolve_report_window(self, now: datetime, period: PeriodType) -> ReportWindow:
        tzinfo = now.tzinfo
        today_start = datetime(now.year, now.month, now.day, tzinfo=tzinfo)

        if period == "yesterday":
            start_at = today_start - timedelta(days=1)
            end_at_exclusive = today_start
            end_display = today_start - timedelta(seconds=1)
            date_label = start_at.date().isoformat()
        elif period == "today":
            start_at = today_start
            end_at_exclusive = now + timedelta(seconds=1)
            end_display = now
            date_label = now.date().isoformat()
        else:  # pragma: no cover - typing 已约束
            raise ValueError(f"unsupported period: {period}")

        return ReportWindow(
            period=period,
            date_label=date_label,
            start_ts=int(start_at.timestamp()),
            end_ts=int(end_at_exclusive.timestamp()),
            time_window=f"{start_at:%Y-%m-%d %H:%M} - {end_display:%Y-%m-%d %H:%M}",
        )

    def _load_template(self) -> str:
        fallback = (
            "{title}\n"
            "统计日期：{date_label}\n"
            "统计范围：{time_window}\n"
            "群组：{group_id}\n\n"
            "【统计总览】\n"
            "- 总发言数：{total_messages}\n"
            "- 参与成员数：{participant_count}\n\n"
            "【活跃成员排行】\n"
            "{active_members_block}\n\n"
            "【整体总结】\n"
            "{overall_summary}\n\n"
            "【热门话题】\n"
            "{group_topics_block}\n\n"
            "【成员兴趣摘要】\n"
            "{member_interest_block}\n\n"
            "【建议 Bot 主动发言】\n"
            "{suggested_bot_reply}\n\n"
            "【语义分析状态】\n"
            "{analysis_notice}\n"
        )

        try:
            return self.template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return fallback
