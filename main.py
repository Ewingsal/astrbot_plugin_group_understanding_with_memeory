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
from .services.storage import JsonMessageStorage


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
        self.storage = JsonMessageStorage(storage_path)
        self.group_origin_store = GroupOriginStore(group_origin_path)
        self.report_cache_store = ReportCacheStore(report_cache_path, cache_version=1)
        self.llm_analysis_service = LLMAnalysisService()
        self.interaction_service = InteractionService()
        self.digest_service = GroupDigestService(
            storage=self.storage,
            llm_analysis_service=self.llm_analysis_service,
            interaction_service=self.interaction_service,
            template_path=Path(__file__).resolve().parent / "templates" / "daily_digest.md.j2",
            report_cache_store=self.report_cache_store,
            cache_version=1,
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
        self._scheduler_started = False

        logger.info(
            "GroupDigestPlugin initialized. data_dir=%s data_dir_scope=%s storage=%s group_origin_store=%s report_cache=%s",
            self._data_dir,
            self._data_dir_scope,
            storage_path,
            group_origin_path,
            report_cache_path,
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
        await self._ensure_scheduler_started()

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
        await self._ensure_scheduler_started()

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
        await self._ensure_scheduler_started()

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
        )
        await self.storage.append_message(record)

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
        await self._ensure_scheduler_started()

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

    def _get_scheduler_start_lock(self) -> asyncio.Lock:
        if self._scheduler_start_lock is None:
            self._scheduler_start_lock = asyncio.Lock()
        return self._scheduler_start_lock

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
            await self.scheduler_service.stop()
        except Exception:
            logger.exception("failed to stop scheduler")
        logger.info("GroupDigestPlugin terminated.")
