from __future__ import annotations

import asyncio
from datetime import datetime

from astrbot.api import logger

from .group_topic_segment_manager import GroupTopicSegmentManager


class TopicLifecycleSweepService:
    """轻量后台任务：定期推进 topic lifecycle（含 prune）。"""

    def __init__(
        self,
        *,
        topic_segment_manager: GroupTopicSegmentManager,
        enabled: bool = True,
        sweep_interval_seconds: int = 60,
    ) -> None:
        self.topic_segment_manager = topic_segment_manager
        self.enabled = bool(enabled)
        self.sweep_interval_seconds = max(10, int(sweep_interval_seconds))

        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._is_running = False

    def start(self) -> bool:
        if not self.enabled:
            logger.info("[group_digest.topic_sweep] disabled by config")
            return True

        if self._task and not self._task.done():
            logger.warning("[group_digest.topic_sweep] already running, skip duplicate start")
            return True

        self._stop_event = None
        self._is_running = True
        try:
            self._task = asyncio.create_task(self._run_loop())
        except RuntimeError:
            self._is_running = False
            logger.exception("[group_digest.topic_sweep] failed to start: no running event loop")
            return False

        logger.info(
            "[group_digest.topic_sweep] started interval_seconds=%d",
            self.sweep_interval_seconds,
        )
        return True

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
                logger.exception("[group_digest.topic_sweep] stop_failed")

        self._task = None
        logger.info("[group_digest.topic_sweep] stopped")

    async def run_once(self, *, now_ts: int | None = None) -> dict[str, int]:
        run_ts = int(now_ts) if now_ts is not None else int(datetime.now().timestamp())
        logger.debug("[group_digest.topic_sweep] trigger now_ts=%d", run_ts)
        return await self.topic_segment_manager.sweep_topics(now_ts=run_ts, enable_prune=True)

    async def _run_loop(self) -> None:
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        while self._is_running:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[group_digest.topic_sweep] loop_error")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.sweep_interval_seconds)
                break
            except asyncio.TimeoutError:
                pass
