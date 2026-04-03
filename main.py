from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from astrbot.api import logger

try:
    from astrbot.api import AstrBotConfig
except Exception:  # pragma: no cover - fallback for older AstrBot versions
    AstrBotConfig = dict  # type: ignore[misc,assignment]

from astrbot.api.event import AstrMessageEvent, filter
try:
    from astrbot.api.star import Context, Star, StarTools, register
except Exception:  # pragma: no cover - fallback for older AstrBot versions
    from astrbot.api.star import Context, Star, register

    StarTools = None  # type: ignore[assignment]

from .services.digest_service import GroupDigestService, PeriodType
from .services.group_origin_store import GroupOriginStore
from .services.interaction_service import InteractionService
from .services.llm_analysis_service import LLMAnalysisService
from .services.message_filters import classify_plugin_owned_message
from .services.models import LLMAnalysisConfig, MessageRecord, SchedulerConfig
from .services.report_cache_store import ReportCacheStore
from .services.scheduler_service import SchedulerRuntimeOptions, ScheduledProactiveService
from .services.semantic_input_builder import SemanticInputBuilder
from .services.group_topic_segment_manager import GroupTopicSegmentManager
from .services.slang_candidate_miner import SlangCandidateMiner
from .services.slang_interpretation_service import SlangInterpretationService
from .services.slang_store import SlangStore
from .services.storage import JsonMessageStorage
from .services.topic_slice_store import TopicSliceStore
from .services.topic_lifecycle_sweep_service import TopicLifecycleSweepService
from .services.embedding import APIEmbeddingBackend, NoopEmbeddingBackend
from .services.embedding_store import NoopEmbeddingStore, QdrantEmbeddingStore


@register("群聊兴趣日报", "Ewingsal", "群聊兴趣日报与主动互动插件（支持定时主动发言与日报缓存）", "0.6.0")
class GroupDigestPlugin(Star):
    def __init__(self, context: Context, config: Optional[AstrBotConfig] = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self._data_dir = self._get_framework_data_dir()
        self._data_dir_scope = self._detect_data_dir_scope(self._data_dir)

        storage_path = self._resolve_storage_path()
        group_origin_path = self._resolve_group_origin_path()
        report_cache_path = self._resolve_report_cache_path()
        topic_slice_root = self._resolve_topic_slice_root()
        slang_store_root = self._resolve_slang_store_root()
        self.storage = JsonMessageStorage(storage_path)
        self.group_origin_store = GroupOriginStore(group_origin_path)
        self.report_cache_store = ReportCacheStore(report_cache_path, cache_version=1)
        self.topic_slice_store = TopicSliceStore(topic_slice_root)
        self.slang_store = SlangStore(slang_store_root)
        self.llm_analysis_service = LLMAnalysisService()
        self.interaction_service = InteractionService()
        self.topic_embedding_backend = self._build_topic_embedding_backend()
        self.topic_embedding_store = self._build_topic_embedding_store()
        self.slang_candidate_miner = SlangCandidateMiner(
            min_term_frequency=self._conf_int(
                "slang_candidate_min_term_frequency",
                2,
                lower=1,
            ),
            min_slice_coverage=self._conf_int(
                "slang_candidate_min_slice_coverage",
                2,
                lower=1,
            ),
            max_candidates=self._conf_int(
                "slang_candidate_max_candidates",
                10,
                lower=1,
                upper=50,
            ),
            current_day_boost=self._conf_float(
                "slang_candidate_current_day_boost",
                0.4,
                lower=0.0,
                upper=3.0,
            ),
        )
        self.slang_interpretation_service = SlangInterpretationService(
            llm_analysis_service=self.llm_analysis_service,
            embedding_backend=self.topic_embedding_backend,
            embedding_store=self.topic_embedding_store,
            slang_store=self.slang_store,
            enable_slang_learning=self._as_bool(
                self._conf_get("enable_slang_learning", False),
                False,
            ),
            slang_retrieval_recent_days=self._conf_int(
                "slang_retrieval_recent_days",
                7,
                lower=1,
                upper=30,
            ),
            slang_retrieval_limit=self._conf_int(
                "slang_retrieval_limit",
                6,
                lower=1,
                upper=50,
            ),
            slang_min_context_items_for_inference=self._conf_int(
                "slang_min_context_items_for_inference",
                2,
                lower=1,
                upper=20,
            ),
            slang_max_inference_per_build=self._conf_int(
                "slang_max_inference_per_build",
                3,
                lower=1,
                upper=20,
            ),
            slang_reinfer_min_evidence_increase=self._conf_int(
                "slang_reinfer_min_evidence_increase",
                2,
                lower=1,
                upper=20,
            ),
        )
        self.group_topic_segment_manager = GroupTopicSegmentManager(
            topic_slice_store=self.topic_slice_store,
            enable_topic_embedding=self._as_bool(
                self._conf_get("enable_topic_embedding", False),
                False,
            ),
            embedding_backend=self.topic_embedding_backend,
            embedding_store=self.topic_embedding_store,
            embedding_model=str(self._conf_get("embedding_model", "")).strip(),
            embedding_version=str(self._conf_get("embedding_version", "v1")).strip() or "v1",
            new_topic_gap_seconds=self._conf_int(
                "new_topic_gap_seconds",
                30 * 60,
                lower=60,
            ),
            topic_close_gap_seconds=self._conf_int(
                "topic_close_gap_seconds",
                20 * 60,
                lower=60,
            ),
            single_message_topic_timeout_seconds=self._conf_int(
                "single_message_topic_timeout_seconds",
                15 * 60,
                lower=60,
            ),
            transfer_similarity_threshold=self._conf_float(
                "transfer_similarity_threshold",
                0.75,
                lower=0.0,
                upper=1.0,
            ),
            transfer_buffer_size=self._conf_int(
                "transfer_buffer_size",
                3,
                lower=1,
            ),
            closed_topic_prune_seconds=self._conf_int(
                "topic_runtime_closed_prune_seconds",
                6 * 60 * 60,
                lower=60,
            ),
        )
        self.topic_lifecycle_sweep_service = TopicLifecycleSweepService(
            topic_segment_manager=self.group_topic_segment_manager,
            enabled=self._as_bool(
                self._conf_get("enable_topic_lifecycle_sweep", True),
                True,
            ),
            sweep_interval_seconds=self._conf_int(
                "topic_lifecycle_sweep_interval_seconds",
                60,
                lower=10,
            ),
        )
        self.semantic_input_builder = SemanticInputBuilder(
            topic_segment_manager=self.group_topic_segment_manager,
            embedding_backend=self.topic_embedding_backend,
            embedding_store=self.topic_embedding_store,
            topic_slice_store=self.topic_slice_store,
            slang_store=self.slang_store,
            slang_candidate_miner=self.slang_candidate_miner,
            slang_interpretation_service=self.slang_interpretation_service,
            enable_topic_slice_contexts=self._as_bool(
                self._conf_get("enable_topic_slice_contexts", True),
                True,
            ),
            enable_topic_slice_retrieval=self._as_bool(
                self._conf_get("enable_topic_slice_retrieval", True),
                True,
            ),
            topic_slice_retrieval_recent_days=self._conf_int(
                "topic_slice_retrieval_recent_days",
                3,
                lower=1,
                upper=30,
            ),
            topic_slice_retrieval_limit=self._conf_int(
                "topic_slice_retrieval_limit",
                5,
                lower=1,
                upper=50,
            ),
            topic_slice_retrieval_query_message_count=self._conf_int(
                "topic_slice_retrieval_query_message_count",
                8,
                lower=1,
                upper=50,
            ),
            enable_slang_contexts=self._as_bool(
                self._conf_get("enable_slang_contexts", False),
                False,
            ),
            max_slang_context_chars=self._conf_int(
                "max_slang_context_chars",
                1200,
                lower=200,
                upper=20000,
            ),
            slang_injection_limit=self._conf_int(
                "slang_injection_limit",
                5,
                lower=1,
                upper=50,
            ),
            slang_recent_days=self._conf_int(
                "slang_recent_days",
                7,
                lower=1,
                upper=30,
            ),
        )
        self.digest_service = GroupDigestService(
            storage=self.storage,
            llm_analysis_service=self.llm_analysis_service,
            interaction_service=self.interaction_service,
            template_path=Path(__file__).resolve().parent / "templates" / "daily_digest.md.j2",
            report_cache_store=self.report_cache_store,
            cache_version=1,
            semantic_input_builder=self.semantic_input_builder,
        )
        self.scheduler_service = ScheduledProactiveService(
            context=self.context,
            digest_service=self.digest_service,
            group_origin_store=self.group_origin_store,
        )
        self.scheduler_config = self._build_scheduler_config()
        self._scheduler_runtime_options = SchedulerRuntimeOptions(
            title_template=str(self._conf_get("title_template", "群聊兴趣日报（{date}）")),
            max_active_members=self._conf_int("max_active_members", 5, lower=1),
            max_topics=self._conf_int("max_topics", 5, lower=1),
        )
        self._scheduler_start_lock: asyncio.Lock | None = None
        self._topic_sweep_start_lock: asyncio.Lock | None = None
        self._scheduler_started = False
        self._topic_sweep_started = False

        logger.info(
            "GroupDigestPlugin initialized. data_dir=%s data_dir_scope=%s storage=%s group_origin_store=%s report_cache=%s topic_slice_store=%s slang_store=%s topic_sweep_enabled=%s topic_sweep_interval=%s topic_embedding_enabled=%s topic_embedding_store_enabled=%s slang_contexts_enabled=%s slang_learning_enabled=%s",
            self._data_dir,
            self._data_dir_scope,
            storage_path,
            group_origin_path,
            report_cache_path,
            topic_slice_root,
            slang_store_root,
            self.topic_lifecycle_sweep_service.enabled,
            self.topic_lifecycle_sweep_service.sweep_interval_seconds,
            self.group_topic_segment_manager.enable_topic_embedding,
            self.topic_embedding_store.enabled,
            self.semantic_input_builder.enable_slang_contexts,
            self.slang_interpretation_service.enable_slang_learning,
        )

    def _conf_get(self, key: str, default: Any) -> Any:
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return default

    def _as_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _build_analysis_config(self) -> LLMAnalysisConfig:
        return LLMAnalysisConfig(
            use_llm_topic_analysis=self._as_bool(self._conf_get("use_llm_topic_analysis", True), True),
            analysis_provider_id=str(self._conf_get("analysis_provider_id", "")).strip(),
            analysis_prompt_template=str(self._conf_get("analysis_prompt_template", "")).strip(),
            interaction_prompt_template=str(self._conf_get("interaction_prompt_template", "")).strip(),
            max_messages_for_analysis=self._conf_int("max_messages_for_analysis", 80, lower=1),
            fallback_to_stats_only=self._as_bool(self._conf_get("fallback_to_stats_only", True), True),
        )

    def _build_scheduler_config(self) -> SchedulerConfig:
        hour = self._conf_int("scheduled_send_hour", 18, lower=0, upper=23)
        minute = self._conf_int("scheduled_send_minute", 0, lower=0, upper=59)
        whitelist = self._as_str_list(self._conf_get("scheduled_group_whitelist", []))
        return SchedulerConfig(
            enable_scheduled_proactive_message=self._as_bool(
                self._conf_get("enable_scheduled_proactive_message", False),
                False,
            ),
            scheduled_send_hour=hour,
            scheduled_send_minute=minute,
            scheduled_mode=str(
                self._conf_get("scheduled_mode", "today_until_scheduled_time")
            ).strip()
            or "today_until_scheduled_time",
            store_group_origin=self._as_bool(self._conf_get("store_group_origin", True), True),
            scheduled_group_whitelist_enabled=self._as_bool(
                self._conf_get("scheduled_group_whitelist_enabled", False),
                False,
            ),
            scheduled_group_whitelist=whitelist,
            scheduled_send_timezone=str(self._conf_get("scheduled_send_timezone", "Asia/Shanghai")).strip()
            or "Asia/Shanghai",
        )

    def _build_topic_embedding_backend(self):
        enable_topic_embedding = self._as_bool(
            self._conf_get("enable_topic_embedding", False),
            False,
        )
        if not enable_topic_embedding:
            logger.info("[group_digest.embedding] embedding_backend=disabled")
            return NoopEmbeddingBackend()

        api_key = str(self._conf_get("embedding_api_key", "")).strip()
        model = str(self._conf_get("embedding_model", "")).strip()
        base_url = str(self._conf_get("embedding_base_url", "")).strip()
        timeout_seconds = self._conf_int("embedding_timeout_seconds", 10, lower=3, upper=120)
        if not api_key or not model:
            logger.warning(
                "[group_digest.embedding] missing embedding_api_key or embedding_model, fallback=noop"
            )
            return NoopEmbeddingBackend()

        return APIEmbeddingBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    def _build_topic_embedding_store(self):
        enabled = self._as_bool(
            self._conf_get("enable_qdrant_embedding_store", False),
            False,
        )
        if not enabled:
            logger.info("[group_digest.embedding_store] qdrant_store=disabled")
            return NoopEmbeddingStore()

        qdrant_url = str(self._conf_get("qdrant_url", "")).strip()
        if not qdrant_url:
            logger.warning(
                "[group_digest.embedding_store] qdrant_url missing, fallback=noop"
            )
            return NoopEmbeddingStore()

        return QdrantEmbeddingStore(
            enabled=True,
            qdrant_url=qdrant_url,
            qdrant_api_key=str(self._conf_get("qdrant_api_key", "")).strip(),
            semantic_unit_collection=str(
                self._conf_get("qdrant_semantic_unit_collection", "group_digest_semantic_units")
            ).strip()
            or "group_digest_semantic_units",
            topic_head_collection=str(
                self._conf_get(
                    "qdrant_topic_head_collection",
                    self._conf_get("qdrant_topic_slice_collection", "group_digest_topic_heads"),
                )
            ).strip()
            or "group_digest_topic_heads",
            topic_slice_collection=str(
                self._conf_get("qdrant_topic_slice_collection", "")
            ).strip(),
            vector_size=self._conf_int("qdrant_vector_size", 1536, lower=1),
            distance_metric=str(self._conf_get("qdrant_distance_metric", "cosine")).strip(),
            prefer_grpc=self._as_bool(self._conf_get("qdrant_prefer_grpc", False), False),
            timeout_seconds=self._conf_int("qdrant_timeout_seconds", 5, lower=1, upper=120),
        )

    def _resolve_storage_path(self) -> Path:
        return self._resolve_data_file_path(
            "storage_path",
            self._default_storage_relative_path("messages.json"),
        )

    def _resolve_group_origin_path(self) -> Path:
        return self._resolve_data_file_path(
            "group_origin_storage_path",
            self._default_storage_relative_path("group_origins.json"),
        )

    def _resolve_report_cache_path(self) -> Path:
        return self._resolve_data_file_path(
            "report_cache_path",
            self._default_storage_relative_path("report_cache.json"),
        )

    def _resolve_topic_slice_root(self) -> Path:
        return self._resolve_data_file_path(
            "topic_slice_storage_path",
            self._default_storage_relative_path("topic_slices"),
        )

    def _resolve_slang_store_root(self) -> Path:
        return self._resolve_data_file_path(
            "slang_store_path",
            self._default_storage_relative_path("slang"),
        )

    def _resolve_data_file_path(self, key: str, default: str) -> Path:
        raw = str(self._conf_get(key, default)).strip()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return self._data_dir / self._normalize_relative_data_path(path)

    def _default_storage_relative_path(self, file_name: str) -> str:
        if self._data_dir_scope == "plugin_dir":
            return file_name
        return f"plugin_data/astrbot_plugin_group_digest/{file_name}"

    def _normalize_relative_data_path(self, path: Path) -> Path:
        if self._data_dir_scope != "plugin_dir":
            return path

        lower_parts = [part.lower() for part in path.parts]
        if len(lower_parts) >= 3 and lower_parts[0] == "plugin_data" and lower_parts[1] == "astrbot_plugin_group_digest":
            normalized = Path(*path.parts[2:])
            logger.info(
                "[group_digest.path] strip_legacy_plugin_data_prefix raw=%s normalized=%s",
                path,
                normalized,
            )
            return normalized
        return path

    def _detect_data_dir_scope(self, data_dir: Path) -> str:
        lower_parts = [part.lower() for part in data_dir.parts]
        if data_dir.name.lower() == "astrbot_plugin_group_digest":
            return "plugin_dir"
        if len(lower_parts) >= 2 and lower_parts[-2] == "plugin_data" and lower_parts[-1] == "astrbot_plugin_group_digest":
            return "plugin_dir"
        return "root_dir"

    def _get_framework_data_dir(self) -> Path:
        # 优先使用官方 StarTools 接口。
        getter = getattr(StarTools, "get_data_dir", None) if StarTools is not None else None
        if callable(getter):
            try:
                value = getter()
                if value:
                    return Path(str(value)).expanduser()
            except Exception as exc:
                logger.warning("failed to read data dir via StarTools.get_data_dir: %s", exc)

        # 兼容部分版本可能挂在 context 上的官方 getter。
        context_getter = getattr(self.context, "get_data_dir", None)
        if callable(context_getter):
            try:
                value = context_getter()
                if value:
                    return Path(str(value)).expanduser()
            except Exception as exc:
                logger.warning("failed to read data dir via context.get_data_dir: %s", exc)

        raise RuntimeError("cannot resolve AstrBot data dir from framework interfaces")

    @filter.command("group_digest")
    async def group_digest(self, event: AstrMessageEvent):
        """手动生成昨日群聊兴趣日报（仅群聊可用）。"""
        async for result in self._handle_digest_command(
            event=event,
            period="yesterday",
            private_hint="/group_digest 仅支持群聊使用，请在群聊中触发。",
        ):
            yield result

    @filter.command("group_digest_today")
    async def group_digest_today(self, event: AstrMessageEvent):
        """手动生成今日（00:00 到当前时刻）群聊兴趣日报（仅群聊可用）。"""
        async for result in self._handle_digest_command(
            event=event,
            period="today",
            private_hint="/group_digest_today 仅支持群聊使用，请在群聊中触发。",
        ):
            yield result

    async def _handle_digest_command(
        self,
        event: AstrMessageEvent,
        period: PeriodType,
        private_hint: str,
    ) -> AsyncGenerator[Any, None]:
        await self._ensure_background_services_started()

        group_id = self._extract_group_id(event)
        if not group_id:
            yield event.plain_result(private_hint)
            return

        now = datetime.now()
        source = "command_group_digest_today" if period == "today" else "command_group_digest"
        try:
            digest_text = await self.digest_service.generate_digest_text_for_period(
                context=self.context,
                event=event,
                group_id=group_id,
                now=now,
                period=period,
                title_template=str(self._conf_get("title_template", "群聊兴趣日报（{date}）")),
                max_active_members=self._conf_int("max_active_members", 5, lower=1),
                max_topics=self._conf_int("max_topics", 5, lower=1),
                analysis_config=self._build_analysis_config(),
                source=source,
            )
        except Exception:
            logger.exception("failed to generate period digest. period=%s", period)
            yield event.plain_result("日报生成失败，请稍后重试。")
            return

        yield event.plain_result(digest_text)

    @filter.command("group_digest_debug_today")
    async def group_digest_debug_today(self, event: AstrMessageEvent):
        """调试命令：统计今天消息，辅助联调归档链路。"""
        await self._ensure_background_services_started()

        group_id = self._extract_group_id(event)
        if not group_id:
            yield event.plain_result("/group_digest_debug_today 仅支持群聊使用，请在群聊中触发。")
            return

        now = datetime.now()
        try:
            debug_text = self.digest_service.generate_today_debug_text(
                group_id=group_id,
                now=now,
                max_active_members=self._conf_int("max_active_members", 5, lower=1),
            )
        except Exception:
            logger.exception("failed to generate today debug stats")
            yield event.plain_result("今日统计生成失败，请稍后重试。")
            return

        yield event.plain_result(debug_text)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def archive_group_message(self, event: AstrMessageEvent):
        """归档群聊消息到本地 JSON，供昨日/今日统计使用。"""
        await self._ensure_background_services_started()

        group_id = self._extract_group_id(event)
        if not group_id:
            return

        content = self._extract_message_text(event)
        if not content:
            return

        sender_id = self._extract_sender_id(event)
        reason = classify_plugin_owned_message(
            text=content,
            sender_id=sender_id,
            bot_sender_ids=self._extract_bot_sender_ids(event),
        )
        if reason:
            logger.info(
                "[group_digest.filter] skip_archive group_id=%s sender_id=%s reason=%s",
                group_id,
                sender_id,
                reason,
            )
            return

        sender_name = self._extract_sender_name(event)
        timestamp = self._extract_timestamp(event)

        record = MessageRecord(
            group_id=group_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=timestamp,
            message_id=self._extract_message_id(event),
        )
        await self.storage.append_message(record)
        try:
            await self.group_topic_segment_manager.ingest_message(record)
        except Exception as exc:
            logger.warning(
                "[group_digest.topic_segment] ingest_failed group_id=%s message_id=%s error=%s",
                group_id,
                record.message_id,
                exc,
            )

        if self.scheduler_config.store_group_origin:
            unified_msg_origin = self._extract_unified_msg_origin(event)
            if unified_msg_origin:
                await self.group_origin_store.upsert_group_origin(
                    group_id=group_id,
                    unified_msg_origin=unified_msg_origin,
                    last_active_at=timestamp,
                )

    async def initialize(self):
        """框架异步初始化钩子：在事件循环就绪后确保 scheduler 启动。"""
        await self._ensure_background_services_started()

    async def _ensure_background_services_started(self) -> None:
        await self._ensure_scheduler_started()
        await self._ensure_topic_sweep_started()

    async def _ensure_scheduler_started(self) -> None:
        if self._scheduler_started:
            return

        lock = self._get_scheduler_start_lock()
        async with lock:
            if self._scheduler_started:
                return

            started = self.scheduler_service.start(
                scheduler_config=self.scheduler_config,
                analysis_config_builder=self._build_analysis_config,
                runtime_options=self._scheduler_runtime_options,
            )
            if started:
                self._scheduler_started = True
                logger.info(
                    "GroupDigestPlugin scheduler start ensured. enabled=%s",
                    self.scheduler_config.enable_scheduled_proactive_message,
                )
                return

            logger.error("GroupDigestPlugin scheduler start failed; will retry on next event.")

    async def _ensure_topic_sweep_started(self) -> None:
        if self._topic_sweep_started:
            return

        lock = self._get_topic_sweep_start_lock()
        async with lock:
            if self._topic_sweep_started:
                return

            started = self.topic_lifecycle_sweep_service.start()
            if started:
                self._topic_sweep_started = True
                logger.info(
                    "GroupDigestPlugin topic lifecycle sweep start ensured. enabled=%s interval=%d",
                    self.topic_lifecycle_sweep_service.enabled,
                    self.topic_lifecycle_sweep_service.sweep_interval_seconds,
                )
                return

            logger.error("GroupDigestPlugin topic lifecycle sweep start failed; will retry on next event.")

    def _get_scheduler_start_lock(self) -> asyncio.Lock:
        if self._scheduler_start_lock is None:
            self._scheduler_start_lock = asyncio.Lock()
        return self._scheduler_start_lock

    def _get_topic_sweep_start_lock(self) -> asyncio.Lock:
        if self._topic_sweep_start_lock is None:
            self._topic_sweep_start_lock = asyncio.Lock()
        return self._topic_sweep_start_lock

    def _extract_group_id(self, event: AstrMessageEvent) -> str:
        """尽量兼容不同平台适配器的 group_id 读取方式。"""
        message_obj = getattr(event, "message_obj", None)
        group_id = getattr(message_obj, "group_id", "") if message_obj else ""
        if group_id:
            return str(group_id)

        getter = getattr(event, "get_group_id", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                # TODO: 确认不同 AstrBot 版本 get_group_id 的行为与返回值类型。
                pass

        return ""

    def _extract_sender_id(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)

        for attr in ("user_id", "id", "qq", "uid"):
            value = getattr(sender, attr, None)
            if value:
                return str(value)

        # TODO: 确认 AstrBot 各平台 sender 字段统一的用户 ID 属性名。
        return "unknown_sender"

    def _extract_sender_name(self, event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_name", None)
        if callable(getter):
            try:
                name = getter()
                if name:
                    return str(name)
            except Exception:
                pass

        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        for attr in ("nickname", "card", "name"):
            value = getattr(sender, attr, None)
            if value:
                return str(value)

        return "未知成员"

    def _extract_timestamp(self, event: AstrMessageEvent) -> int:
        message_obj = getattr(event, "message_obj", None)
        ts = getattr(message_obj, "timestamp", None)
        if ts is None:
            return int(datetime.now().timestamp())

        try:
            return int(ts)
        except (TypeError, ValueError):
            return int(datetime.now().timestamp())

    def _extract_message_id(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        for attr in ("message_id", "msg_id", "id"):
            value = getattr(message_obj, attr, None)
            if value:
                return str(value)

        for attr in ("message_id", "msg_id", "id"):
            value = getattr(event, attr, None)
            if value:
                return str(value)

        getter = getattr(event, "get_message_id", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                # TODO: 确认不同 AstrBot 版本 get_message_id 的行为。
                pass

        return ""

    def _extract_message_text(self, event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", "")
        if text:
            return str(text)

        message_obj = getattr(event, "message_obj", None)
        text = getattr(message_obj, "message_str", "")
        return str(text) if text else ""

    def _extract_unified_msg_origin(self, event: AstrMessageEvent) -> str:
        value = getattr(event, "unified_msg_origin", None)
        if value:
            return str(value)

        getter = getattr(event, "get_unified_msg_origin", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                # TODO: 确认不同 AstrBot 版本 get_unified_msg_origin 的行为。
                pass

        return ""

    def _extract_bot_sender_ids(self, event: AstrMessageEvent) -> set[str]:
        values: set[str] = set()

        for attr in ("self_id", "bot_id", "account_id"):
            val = getattr(self.context, attr, None)
            if val:
                values.add(str(val))

        for attr in ("self_id", "bot_id"):
            val = getattr(event, attr, None)
            if val:
                values.add(str(val))

        getter = getattr(event, "get_self_id", None)
        if callable(getter):
            try:
                val = getter()
                if val:
                    values.add(str(val))
            except Exception:
                pass

        message_obj = getattr(event, "message_obj", None)
        for attr in ("self_id", "bot_id"):
            val = getattr(message_obj, attr, None)
            if val:
                values.add(str(val))

        return values

    def _conf_int(
        self,
        key: str,
        default: int,
        *,
        lower: int | None = None,
        upper: int | None = None,
    ) -> int:
        value = self._conf_get(key, default)
        try:
            num = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.config] invalid_int key=%s value=%r fallback=%d",
                key,
                value,
                default,
            )
            num = default

        if lower is not None and num < lower:
            logger.warning(
                "[group_digest.config] int_below_lower_bound key=%s value=%d lower=%d fallback=%d",
                key,
                num,
                lower,
                default,
            )
            num = lower
        if upper is not None and num > upper:
            logger.warning(
                "[group_digest.config] int_above_upper_bound key=%s value=%d upper=%d fallback=%d",
                key,
                num,
                upper,
                default,
            )
            num = upper
        return num

    def _conf_float(
        self,
        key: str,
        default: float,
        *,
        lower: float | None = None,
        upper: float | None = None,
    ) -> float:
        value = self._conf_get(key, default)
        try:
            num = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "[group_digest.config] invalid_float key=%s value=%r fallback=%s",
                key,
                value,
                default,
            )
            num = float(default)

        if lower is not None and num < lower:
            logger.warning(
                "[group_digest.config] float_below_lower_bound key=%s value=%s lower=%s fallback=%s",
                key,
                num,
                lower,
                lower,
            )
            num = float(lower)
        if upper is not None and num > upper:
            logger.warning(
                "[group_digest.config] float_above_upper_bound key=%s value=%s upper=%s fallback=%s",
                key,
                num,
                upper,
                upper,
            )
            num = float(upper)
        return num

    def _as_str_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            rows = value
        elif isinstance(value, str):
            rows = [part.strip() for part in value.split(",")]
        else:
            rows = []

        result: list[str] = []
        for row in rows:
            text = str(row).strip()
            if text:
                result.append(text)
        return result

    async def terminate(self):
        """插件卸载/停用时调用。"""
        try:
            await self.topic_lifecycle_sweep_service.stop()
            self._topic_sweep_started = False
        except Exception:
            logger.exception("failed to stop topic lifecycle sweep")
        try:
            await self.scheduler_service.stop()
            self._scheduler_started = False
        except Exception:
            logger.exception("failed to stop scheduler")
        logger.info("GroupDigestPlugin terminated.")
