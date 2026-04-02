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
from .incremental_update_service import EffectiveMessageState, IncrementalUpdateService
from .llm_analysis_service import LLMAnalysisService
from .message_filters import filter_effective_messages
from .models import DigestReport, LLMAnalysisConfig, LLMSemanticResult, MemberDigest, MessageRecord
from .report_cache_store import ReportCacheRecord, ReportCacheStore
from .semantic_input_builder import SemanticInputBuilder, SemanticInputMaterial
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
    build_path: str = "full_rebuild"
    delta_message_count: int = 0
    incremental_round: int = 0
    semantic_input_source: str = ""
    retrieved_topic_slice_count: int = 0
    current_day_topic_slice_count: int = 0
    retrieval_enabled: bool = False
    retrieval_degraded: bool = False
    retrieval_query_chars: int = 0


@dataclass(frozen=True)
class CacheDecision:
    strategy: Literal["cache_hit", "incremental_update", "full_rebuild"]
    reason: str
    delta_messages: list[MessageRecord]
    incremental_round: int = 0


class GroupDigestService:
    INCREMENTAL_SUPPORTED_MODES = {"today", "scheduled"}
    INCREMENTAL_MAX_DELTA_MESSAGES = 20
    INCREMENTAL_MAX_DELTA_RATIO = 0.5
    INCREMENTAL_MAX_ROUNDS = 3

    def __init__(
        self,
        storage: JsonMessageStorage,
        llm_analysis_service: LLMAnalysisService,
        interaction_service: InteractionService,
        template_path: Path,
        report_cache_store: ReportCacheStore | None = None,
        cache_version: int = 1,
        semantic_input_builder: SemanticInputBuilder | None = None,
    ):
        self.storage = storage
        self.llm_analysis_service = llm_analysis_service
        self.interaction_service = interaction_service
        self.template_path = template_path
        self.report_cache_store = report_cache_store
        self.cache_version = int(cache_version)
        self.incremental_service = IncrementalUpdateService()
        self.semantic_input_builder = semantic_input_builder or SemanticInputBuilder()

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
        ordered_effective_messages = self.incremental_service.sort_messages(effective_messages)
        effective_state = self.incremental_service.build_effective_state(ordered_effective_messages)
        message_count = effective_state.message_count
        last_message_ts = effective_state.last_message_ts
        last_message_fingerprint = effective_state.last_message_fingerprint
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

        full_window_semantic_material = await self.semantic_input_builder.build_for_full_window(
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            mode=cache_mode,
            effective_messages=ordered_effective_messages,
            max_messages_for_analysis=config.max_messages_for_analysis,
            context=context,
            event=event,
            analysis_provider_id=config.analysis_provider_id,
        )

        expected_provider_id, expected_provider_source, expected_provider_err = await self._resolve_expected_provider_id(
            context=context,
            event=event,
            config=config,
        )

        cache_record: ReportCacheRecord | None = None
        full_rebuild_reason = "no_cache_store"
        fallback_from_incremental = False

        if self.report_cache_store is not None:
            cache_record = self.report_cache_store.get_record(
                group_id=group_id,
                date=window.date_label,
                mode=cache_mode,
            )
            decision = self._decide_cache_strategy(
                cache_record=cache_record,
                cache_mode=cache_mode,
                effective_messages=ordered_effective_messages,
                effective_state=effective_state,
                expected_provider_id=expected_provider_id,
                expected_provider_err=expected_provider_err,
                max_messages_for_analysis=config.max_messages_for_analysis,
                prompt_signature=prompt_signature,
                window_start=window.start_ts,
                use_llm_topic_analysis=config.use_llm_topic_analysis,
                semantic_material=full_window_semantic_material,
            )

            if decision.strategy == "cache_hit" and cache_record is not None:
                cached_report = self._report_from_payload(cache_record.report)
                if cached_report is not None:
                    logger.info(
                        "[group_digest.cache] cache_hit group_id=%s date=%s mode=%s reason=%s effective_message_count=%s effective_last_message_ts=%s effective_last_message_fingerprint=%s provider=%s semantic_source=%s topic_slice_count=%d topic_slice_truncated=%s tail_count=%d",
                        group_id,
                        window.date_label,
                        cache_mode,
                        decision.reason,
                        message_count,
                        last_message_ts,
                        last_message_fingerprint,
                        expected_provider_id or "-",
                        full_window_semantic_material.source,
                        full_window_semantic_material.topic_slice_selected_count,
                        "true" if full_window_semantic_material.topic_slice_truncated else "false",
                        full_window_semantic_material.selected_message_count,
                    )
                    return (
                        cached_report,
                        ReportBuildMetrics(
                            load_messages_ms=load_messages_ms,
                            aggregate_stats_ms=0,
                            llm_analysis_ms=0,
                            build_path="cache_hit",
                            delta_message_count=0,
                            incremental_round=max(0, cache_record.incremental_round),
                            semantic_input_source=full_window_semantic_material.source,
                            retrieved_topic_slice_count=full_window_semantic_material.retrieved_topic_slice_count,
                            current_day_topic_slice_count=full_window_semantic_material.current_day_topic_slice_count,
                            retrieval_enabled=bool(full_window_semantic_material.retrieval_enabled),
                            retrieval_degraded=bool(full_window_semantic_material.retrieval_degraded),
                            retrieval_query_chars=full_window_semantic_material.retrieval_query_chars,
                        ),
                    )
                logger.warning(
                    "[group_digest.cache] cached_report_invalid group_id=%s date=%s mode=%s fallback_to_full_rebuild=true",
                    group_id,
                    window.date_label,
                    cache_mode,
                )
                full_rebuild_reason = "cached_report_invalid"
            elif decision.strategy == "incremental_update" and cache_record is not None:
                logger.info(
                    "[group_digest.cache] incremental_update group_id=%s date=%s mode=%s reason=%s delta_message_count=%d effective_message_count=%d incremental_round=%d provider=%s semantic_source=%s topic_slice_count=%d topic_slice_truncated=%s tail_count=%d",
                    group_id,
                    window.date_label,
                    cache_mode,
                    decision.reason,
                    len(decision.delta_messages),
                    message_count,
                    decision.incremental_round,
                    expected_provider_id or "-",
                    full_window_semantic_material.source,
                    full_window_semantic_material.topic_slice_selected_count,
                    "true" if full_window_semantic_material.topic_slice_truncated else "false",
                    full_window_semantic_material.selected_message_count,
                )
                try:
                    incremental_result = await self._build_report_with_incremental_update(
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
                        cache_record=cache_record,
                        delta_messages=decision.delta_messages,
                        load_messages_ms=load_messages_ms,
                        incremental_round=decision.incremental_round,
                    )
                except Exception as exc:
                    logger.warning(
                        "[group_digest.cache] incremental_update_exception group_id=%s date=%s mode=%s error=%s",
                        group_id,
                        window.date_label,
                        cache_mode,
                        exc,
                    )
                    incremental_result = None
                if incremental_result is not None:
                    report, metrics, updated_stats_state, updated_semantic_state, incremental_semantic_material = incremental_result
                    if report is not None:
                        await self._write_cache_record(
                            report=report,
                            cache_mode=cache_mode,
                            group_id=group_id,
                            window=window,
                            source=source,
                            config=config,
                            prompt_signature=prompt_signature,
                            provider_id=report.analysis_provider_id or expected_provider_id,
                            effective_state=effective_state,
                            stats_state=updated_stats_state,
                            semantic_state=updated_semantic_state,
                            incremental_round=decision.incremental_round,
                            semantic_material=incremental_semantic_material,
                        )
                    return report, metrics

                full_rebuild_reason = "incremental_update_failed"
                fallback_from_incremental = True
            else:
                full_rebuild_reason = decision.reason

            logger.info(
                "[group_digest.cache] full_rebuild group_id=%s date=%s mode=%s reason=%s delta_message_count=%d effective_message_count=%d incremental_round=%d provider=%s whether_fallback_to_full_rebuild=%s semantic_source=%s topic_slice_count=%d topic_slice_truncated=%s tail_count=%d",
                group_id,
                window.date_label,
                cache_mode,
                full_rebuild_reason,
                len(decision.delta_messages),
                message_count,
                decision.incremental_round,
                expected_provider_id or "-",
                "true" if fallback_from_incremental else "false",
                full_window_semantic_material.source,
                full_window_semantic_material.topic_slice_selected_count,
                "true" if full_window_semantic_material.topic_slice_truncated else "false",
                full_window_semantic_material.selected_message_count,
            )
        else:
            logger.info(
                "[group_digest.cache] full_rebuild group_id=%s date=%s mode=%s reason=%s delta_message_count=%d effective_message_count=%d incremental_round=0 provider=%s whether_fallback_to_full_rebuild=false semantic_source=%s topic_slice_count=%d topic_slice_truncated=%s tail_count=%d",
                group_id,
                window.date_label,
                cache_mode,
                full_rebuild_reason,
                0,
                message_count,
                expected_provider_id or "-",
                full_window_semantic_material.source,
                full_window_semantic_material.topic_slice_selected_count,
                "true" if full_window_semantic_material.topic_slice_truncated else "false",
                full_window_semantic_material.selected_message_count,
            )

        report, metrics, rebuilt_stats_state, rebuilt_semantic_state = await self._build_report_without_cache(
            context=context,
            event=event,
            group_id=group_id,
            window=window,
            period=period,
            cache_mode=cache_mode,
            max_active_members=max_active_members,
            max_topics=max_topics,
            analysis_config=config,
            expected_provider_id=expected_provider_id,
            expected_provider_source=expected_provider_source,
            effective_messages=ordered_effective_messages,
            load_messages_ms=load_messages_ms,
            prebuilt_semantic_material=full_window_semantic_material,
        )

        if report is not None and self.report_cache_store is not None:
            await self._write_cache_record(
                report=report,
                cache_mode=cache_mode,
                group_id=group_id,
                window=window,
                source=source,
                config=config,
                prompt_signature=prompt_signature,
                provider_id=report.analysis_provider_id or expected_provider_id,
                effective_state=effective_state,
                stats_state=rebuilt_stats_state,
                semantic_state=rebuilt_semantic_state,
                incremental_round=0,
                semantic_material=full_window_semantic_material,
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
        cache_mode: ReportMode,
        max_active_members: int,
        max_topics: int,
        analysis_config: LLMAnalysisConfig,
        expected_provider_id: str,
        expected_provider_source: str,
        effective_messages: list[MessageRecord],
        load_messages_ms: int,
        prebuilt_semantic_material: SemanticInputMaterial | None = None,
    ) -> tuple[DigestReport | None, ReportBuildMetrics, dict[str, Any], dict[str, Any]]:
        if not effective_messages:
            return (
                None,
                ReportBuildMetrics(
                    load_messages_ms=load_messages_ms,
                    aggregate_stats_ms=0,
                    llm_analysis_ms=0,
                    build_path="no_messages",
                ),
                {},
                {},
            )

        aggregate_start = perf_counter()
        stats_state = self.incremental_service.build_stats_state_from_messages(effective_messages)
        report = self._build_stats_report_from_state(
            period=period,
            group_id=group_id,
            window=window,
            stats_state=stats_state,
            max_active_members=max_active_members,
        )
        aggregate_stats_ms = int((perf_counter() - aggregate_start) * 1000)

        semantic_material = prebuilt_semantic_material or await self.semantic_input_builder.build_for_full_window(
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            mode=cache_mode,
            effective_messages=effective_messages,
            max_messages_for_analysis=analysis_config.max_messages_for_analysis,
            context=context,
            event=event,
            analysis_provider_id=analysis_config.analysis_provider_id,
        )

        llm_start = perf_counter()
        outcome = await self.llm_analysis_service.analyze(
            context=context,
            event=event,
            config=analysis_config,
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            messages=semantic_material.messages,
            active_members=report.active_members,
            max_topics=max_topics,
            resolved_provider_id=expected_provider_id if expected_provider_id else None,
            resolved_provider_source=expected_provider_source,
            topic_slice_contexts=semantic_material.topic_slice_contexts,
            semantic_input_source=semantic_material.source,
        )
        llm_analysis_ms = int((perf_counter() - llm_start) * 1000)

        semantic_state = self._apply_analysis_outcome_to_report(
            report=report,
            outcome=outcome,
            analysis_config=analysis_config,
        )

        return (
            report,
            ReportBuildMetrics(
                load_messages_ms=load_messages_ms,
                aggregate_stats_ms=aggregate_stats_ms,
                llm_analysis_ms=llm_analysis_ms,
                build_path="full_rebuild",
                semantic_input_source=semantic_material.source,
                retrieved_topic_slice_count=semantic_material.retrieved_topic_slice_count,
                current_day_topic_slice_count=semantic_material.current_day_topic_slice_count,
                retrieval_enabled=bool(semantic_material.retrieval_enabled),
                retrieval_degraded=bool(semantic_material.retrieval_degraded),
                retrieval_query_chars=semantic_material.retrieval_query_chars,
            ),
            stats_state,
            semantic_state,
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

    def _decide_cache_strategy(
        self,
        *,
        cache_record: ReportCacheRecord | None,
        cache_mode: ReportMode,
        effective_messages: list[MessageRecord],
        effective_state: EffectiveMessageState,
        expected_provider_id: str,
        expected_provider_err: str,
        max_messages_for_analysis: int,
        prompt_signature: str,
        window_start: int,
        use_llm_topic_analysis: bool,
        semantic_material: SemanticInputMaterial,
    ) -> CacheDecision:
        if cache_record is None:
            return CacheDecision(
                strategy="full_rebuild",
                reason="no_cache_entry",
                delta_messages=[],
                incremental_round=0,
            )

        if cache_record.cache_version != self.cache_version:
            return CacheDecision(
                strategy="full_rebuild",
                reason="cache_version_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if cache_record.window_start != window_start:
            return CacheDecision(
                strategy="full_rebuild",
                reason="window_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if cache_record.max_messages_for_analysis != max_messages_for_analysis:
            return CacheDecision(
                strategy="full_rebuild",
                reason="max_messages_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if cache_record.prompt_signature != prompt_signature:
            return CacheDecision(
                strategy="full_rebuild",
                reason="prompt_signature_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if expected_provider_err:
            return CacheDecision(
                strategy="full_rebuild",
                reason="provider_validation_failed",
                delta_messages=[],
                incremental_round=0,
            )
        if expected_provider_id and cache_record.provider_id != expected_provider_id:
            return CacheDecision(
                strategy="full_rebuild",
                reason="provider_changed",
                delta_messages=[],
                incremental_round=0,
            )

        cached_slice_signature = str(cache_record.topic_slice_signature or "").strip()
        if cached_slice_signature != semantic_material.topic_slice_signature:
            return CacheDecision(
                strategy="full_rebuild",
                reason="topic_slice_signature_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if int(cache_record.topic_slice_count) != semantic_material.topic_slice_selected_count:
            return CacheDecision(
                strategy="full_rebuild",
                reason="topic_slice_count_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if int(cache_record.topic_slice_selected_chars) != semantic_material.topic_slice_selected_chars:
            return CacheDecision(
                strategy="full_rebuild",
                reason="topic_slice_chars_changed",
                delta_messages=[],
                incremental_round=0,
            )
        if bool(cache_record.topic_slice_truncated) != bool(semantic_material.topic_slice_truncated):
            return CacheDecision(
                strategy="full_rebuild",
                reason="topic_slice_truncation_changed",
                delta_messages=[],
                incremental_round=0,
            )

        cached_effective_count = self._cached_effective_count(cache_record)
        cached_effective_last_ts = self._cached_effective_last_ts(cache_record)
        cached_effective_last_fingerprint = self._cached_effective_last_fingerprint(cache_record)

        if (
            cached_effective_count == effective_state.message_count
            and cached_effective_last_ts == effective_state.last_message_ts
            and cached_effective_last_fingerprint == effective_state.last_message_fingerprint
        ):
            return CacheDecision(
                strategy="cache_hit",
                reason="no_new_effective_messages",
                delta_messages=[],
                incremental_round=max(0, cache_record.incremental_round),
            )

        if cache_mode not in self.INCREMENTAL_SUPPORTED_MODES:
            return CacheDecision(
                strategy="full_rebuild",
                reason="mode_not_incremental",
                delta_messages=[],
                incremental_round=0,
            )

        if effective_state.message_count < cached_effective_count:
            return CacheDecision(
                strategy="full_rebuild",
                reason="effective_message_count_decreased",
                delta_messages=[],
                incremental_round=0,
            )

        boundary_reason, delta_messages = self.incremental_service.locate_delta_messages(
            messages=effective_messages,
            checkpoint_last_message_ts=cached_effective_last_ts,
            checkpoint_last_message_fingerprint=cached_effective_last_fingerprint,
        )
        if boundary_reason:
            return CacheDecision(
                strategy="full_rebuild",
                reason=boundary_reason,
                delta_messages=[],
                incremental_round=0,
            )

        delta_count = len(delta_messages)
        if delta_count <= 0:
            return CacheDecision(
                strategy="cache_hit",
                reason="no_new_effective_messages",
                delta_messages=[],
                incremental_round=max(0, cache_record.incremental_round),
            )

        if delta_count > self.INCREMENTAL_MAX_DELTA_MESSAGES:
            return CacheDecision(
                strategy="full_rebuild",
                reason="delta_messages_too_many",
                delta_messages=delta_messages,
                incremental_round=0,
            )

        delta_ratio = delta_count / max(1, effective_state.message_count)
        if delta_ratio > self.INCREMENTAL_MAX_DELTA_RATIO:
            return CacheDecision(
                strategy="full_rebuild",
                reason="delta_ratio_too_high",
                delta_messages=delta_messages,
                incremental_round=0,
            )

        next_incremental_round = max(0, cache_record.incremental_round) + 1
        if next_incremental_round > self.INCREMENTAL_MAX_ROUNDS:
            return CacheDecision(
                strategy="full_rebuild",
                reason="incremental_round_limit_exceeded",
                delta_messages=delta_messages,
                incremental_round=next_incremental_round,
            )

        if self.incremental_service.normalize_stats_state(cache_record.stats_state) is None:
            return CacheDecision(
                strategy="full_rebuild",
                reason="checkpoint_stats_state_invalid",
                delta_messages=delta_messages,
                incremental_round=next_incremental_round,
            )

        # 语义分析开启时，增量语义更新需要上一版语义状态。
        if use_llm_topic_analysis and not self._semantic_state_from_cache(cache_record):
            return CacheDecision(
                strategy="full_rebuild",
                reason="checkpoint_semantic_state_missing",
                delta_messages=delta_messages,
                incremental_round=next_incremental_round,
            )

        return CacheDecision(
            strategy="incremental_update",
            reason="new_effective_messages_detected",
            delta_messages=delta_messages,
            incremental_round=next_incremental_round,
        )

    def _cached_effective_count(self, cache_record: ReportCacheRecord) -> int:
        value = self._safe_int(
            cache_record.effective_message_count,
            field="cache.effective_message_count",
            default=cache_record.message_count,
        )
        if value < 0:
            return 0
        return value

    def _cached_effective_last_ts(self, cache_record: ReportCacheRecord) -> int:
        value = self._safe_int(
            cache_record.effective_last_message_ts,
            field="cache.effective_last_message_ts",
            default=cache_record.last_message_timestamp,
        )
        if value < 0:
            return 0
        return value

    def _cached_effective_last_fingerprint(self, cache_record: ReportCacheRecord) -> str:
        value = str(cache_record.effective_last_message_fingerprint or "").strip()
        if value:
            return value

        # 兼容旧缓存：如果没有 fingerprint，保守回退全量重算。
        return ""

    async def _build_report_with_incremental_update(
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
        cache_record: ReportCacheRecord,
        delta_messages: list[MessageRecord],
        load_messages_ms: int,
        incremental_round: int,
    ) -> tuple[DigestReport, ReportBuildMetrics, dict[str, Any], dict[str, Any], SemanticInputMaterial] | None:
        aggregate_start = perf_counter()
        updated_stats_state = self.incremental_service.apply_delta_to_stats_state(
            base_state=cache_record.stats_state,
            delta_messages=delta_messages,
        )
        if updated_stats_state is None:
            logger.warning(
                "[group_digest.cache] incremental_update_failed group_id=%s date=%s mode=%s reason=invalid_stats_state",
                group_id,
                window.date_label,
                cache_record.mode,
            )
            return None

        report = self._build_stats_report_from_state(
            period=period,
            group_id=group_id,
            window=window,
            stats_state=updated_stats_state,
            max_active_members=max_active_members,
        )
        aggregate_stats_ms = int((perf_counter() - aggregate_start) * 1000)

        previous_semantic_state = self._semantic_state_from_cache(cache_record)
        semantic_material = await self.semantic_input_builder.build_for_incremental(
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            mode=cache_record.mode,
            delta_messages=delta_messages,
            max_messages_for_analysis=analysis_config.max_messages_for_analysis,
            context=context,
            event=event,
            analysis_provider_id=analysis_config.analysis_provider_id,
        )
        llm_start = perf_counter()
        outcome = await self.llm_analysis_service.analyze_incremental(
            context=context,
            event=event,
            config=analysis_config,
            group_id=group_id,
            date_label=window.date_label,
            time_window=window.time_window,
            delta_messages=semantic_material.messages,
            previous_semantic_state=previous_semantic_state,
            updated_stats_state=updated_stats_state,
            max_topics=max_topics,
            resolved_provider_id=expected_provider_id if expected_provider_id else None,
            resolved_provider_source=expected_provider_source,
            topic_slice_contexts=semantic_material.topic_slice_contexts,
            semantic_input_source=semantic_material.source,
        )
        llm_analysis_ms = int((perf_counter() - llm_start) * 1000)

        if outcome.error:
            logger.warning(
                "[group_digest.cache] incremental_update_failed group_id=%s date=%s mode=%s reason=llm_incremental_failed error=%s",
                group_id,
                window.date_label,
                cache_record.mode,
                outcome.error,
            )
            return None

        semantic_state = self._apply_analysis_outcome_to_report(
            report=report,
            outcome=outcome,
            analysis_config=analysis_config,
        )

        return (
            report,
            ReportBuildMetrics(
                load_messages_ms=load_messages_ms,
                aggregate_stats_ms=aggregate_stats_ms,
                llm_analysis_ms=llm_analysis_ms,
                build_path="incremental_update",
                delta_message_count=len(delta_messages),
                incremental_round=incremental_round,
                semantic_input_source=semantic_material.source,
                retrieved_topic_slice_count=semantic_material.retrieved_topic_slice_count,
                current_day_topic_slice_count=semantic_material.current_day_topic_slice_count,
                retrieval_enabled=bool(semantic_material.retrieval_enabled),
                retrieval_degraded=bool(semantic_material.retrieval_degraded),
                retrieval_query_chars=semantic_material.retrieval_query_chars,
            ),
            updated_stats_state,
            semantic_state,
            semantic_material,
        )

    async def _write_cache_record(
        self,
        *,
        report: DigestReport,
        cache_mode: ReportMode,
        group_id: str,
        window: ReportWindow,
        source: str,
        config: LLMAnalysisConfig,
        prompt_signature: str,
        provider_id: str,
        effective_state: EffectiveMessageState,
        stats_state: dict[str, Any],
        semantic_state: dict[str, Any],
        incremental_round: int,
        semantic_material: SemanticInputMaterial,
    ) -> None:
        if self.report_cache_store is None:
            return

        record = ReportCacheRecord(
            group_id=group_id,
            date=window.date_label,
            mode=cache_mode,
            window_start=window.start_ts,
            window_end=window.end_ts,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            last_message_timestamp=effective_state.last_message_ts,
            message_count=effective_state.message_count,
            provider_id=provider_id,
            analysis_provider_notice=report.analysis_notice,
            max_messages_for_analysis=config.max_messages_for_analysis,
            prompt_signature=prompt_signature,
            cache_version=self.cache_version,
            source=source,
            report=self._report_to_payload(report),
            effective_message_count=effective_state.message_count,
            effective_last_message_ts=effective_state.last_message_ts,
            effective_last_message_fingerprint=effective_state.last_message_fingerprint,
            stats_state=stats_state if isinstance(stats_state, dict) else {},
            semantic_state=semantic_state if isinstance(semantic_state, dict) else {},
            incremental_round=max(0, incremental_round),
            semantic_input_source=semantic_material.source,
            topic_slice_signature=semantic_material.topic_slice_signature,
            topic_slice_count=semantic_material.topic_slice_selected_count,
            topic_slice_total_chars=semantic_material.topic_slice_total_chars,
            topic_slice_selected_chars=semantic_material.topic_slice_selected_chars,
            topic_slice_truncated=bool(semantic_material.topic_slice_truncated),
        )
        await self.report_cache_store.upsert_record(record)
        logger.info(
            "[group_digest.cache] cache_write group_id=%s date=%s mode=%s source=%s effective_message_count=%s effective_last_message_ts=%s incremental_round=%d semantic_source=%s topic_slice_count=%d topic_slice_truncated=%s tail_count=%d",
            group_id,
            window.date_label,
            cache_mode,
            source,
            effective_state.message_count,
            effective_state.last_message_ts,
            max(0, incremental_round),
            semantic_material.source,
            semantic_material.topic_slice_selected_count,
            "true" if semantic_material.topic_slice_truncated else "false",
            semantic_material.selected_message_count,
        )

    def _build_prompt_signature(self, *, config: LLMAnalysisConfig, max_topics: int) -> str:
        semantic_builder_meta = self.semantic_input_builder.describe_extension_point()
        payload = {
            "analysis_prompt_template": config.analysis_prompt_template.strip(),
            "interaction_prompt_template": config.interaction_prompt_template.strip(),
            "use_llm_topic_analysis": config.use_llm_topic_analysis,
            "max_topics": max_topics,
            "semantic_input_builder": {
                "topic_slice_contexts_enabled": bool(semantic_builder_meta.get("topic_slice_contexts_enabled", False)),
                "topic_slice_context_char_guard": int(semantic_builder_meta.get("topic_slice_context_char_guard", 0) or 0),
            },
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

    def _semantic_state_from_semantic(self, semantic: LLMSemanticResult | None) -> dict[str, Any]:
        if semantic is None:
            return {}
        return {
            "group_topics": [str(item).strip() for item in semantic.group_topics if str(item).strip()],
            "member_interests": {
                str(name).strip(): str(summary).strip()
                for name, summary in semantic.member_interests.items()
                if str(name).strip() and str(summary).strip()
            },
            "overall_summary": str(semantic.overall_summary).strip(),
            "suggested_bot_reply": str(semantic.suggested_bot_reply).strip(),
        }

    def _semantic_state_from_cache(self, cache_record: ReportCacheRecord) -> dict[str, Any]:
        semantic_state = cache_record.semantic_state
        if isinstance(semantic_state, dict):
            parsed = self._parse_semantic_state_dict(semantic_state)
            if parsed:
                return parsed

        report = self._report_from_payload(cache_record.report)
        if report is not None and report.llm_semantic is not None:
            parsed = self._semantic_state_from_semantic(report.llm_semantic)
            if parsed:
                return parsed
        return {}

    def _parse_semantic_state_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        topics_raw = data.get("group_topics", [])
        topics = [
            str(item).strip()
            for item in topics_raw
            if str(item).strip()
        ] if isinstance(topics_raw, list) else []

        interests_raw = data.get("member_interests", {})
        interests = (
            {
                str(name).strip(): str(summary).strip()
                for name, summary in interests_raw.items()
                if str(name).strip() and str(summary).strip()
            }
            if isinstance(interests_raw, dict)
            else {}
        )

        overall_summary = str(data.get("overall_summary", "")).strip()
        suggested = str(data.get("suggested_bot_reply", "")).strip()
        if not topics or not overall_summary or not suggested:
            return {}
        return {
            "group_topics": topics,
            "member_interests": interests,
            "overall_summary": overall_summary,
            "suggested_bot_reply": suggested,
        }

    def _apply_analysis_outcome_to_report(
        self,
        *,
        report: DigestReport,
        outcome: Any,
        analysis_config: LLMAnalysisConfig,
    ) -> dict[str, Any]:
        if outcome.semantic is not None:
            outcome.semantic.suggested_bot_reply = self.interaction_service.finalize_suggested_reply(
                outcome.semantic.suggested_bot_reply
            )
            report.llm_semantic = outcome.semantic
            report.stats_only = False
            report.analysis_notice = outcome.notice
            report.analysis_provider_id = outcome.provider_id
            return self._semantic_state_from_semantic(outcome.semantic)

        if outcome.error:
            report.analysis_provider_id = outcome.provider_id
            if analysis_config.fallback_to_stats_only:
                report.stats_only = True
                report.analysis_notice = f"语义分析失败，已降级为仅统计：{outcome.error}"
            else:
                report.stats_only = True
                report.analysis_notice = f"[ERROR] 语义分析失败：{outcome.error}"
            return {}

        report.stats_only = True
        report.analysis_notice = outcome.notice or "语义分析已关闭，当前仅展示统计结果。"
        report.analysis_provider_id = outcome.provider_id
        return {}

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
        stats_state = self.incremental_service.build_stats_state_from_messages(messages)
        return self._build_stats_report_from_state(
            period=period,
            group_id=group_id,
            window=window,
            stats_state=stats_state,
            max_active_members=max_active_members,
        )

    def _build_stats_report_from_state(
        self,
        *,
        period: PeriodType,
        group_id: str,
        window: ReportWindow,
        stats_state: Any,
        max_active_members: int,
    ) -> DigestReport:
        normalized = self.incremental_service.normalize_stats_state(stats_state)
        if normalized is None:
            normalized = {
                "total_messages": 0,
                "participant_count": 0,
                "member_message_counts": {},
            }
        total_messages = self._safe_int(
            normalized.get("total_messages", 0),
            field="stats_state.total_messages",
            default=0,
        )
        participant_count = self._safe_int(
            normalized.get("participant_count", 0),
            field="stats_state.participant_count",
            default=0,
        )
        active_members = self.incremental_service.build_active_members_from_stats_state(
            state=normalized,
            max_active_members=max_active_members,
        )
        return DigestReport(
            period=period,
            date_label=window.date_label,
            time_window=window.time_window,
            group_id=group_id,
            total_messages=max(0, total_messages),
            participant_count=max(0, participant_count),
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
