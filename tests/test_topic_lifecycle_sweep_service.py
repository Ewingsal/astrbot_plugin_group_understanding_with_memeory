from __future__ import annotations

import asyncio

from astrbot_plugin_group_digest.services.topic_lifecycle_sweep_service import TopicLifecycleSweepService


class _StubTopicManager:
    def __init__(self):
        self.calls: list[dict] = []

    async def sweep_topics(self, *, now_ts=None, group_id=None, date_label=None, enable_prune=True):
        call = {
            "now_ts": now_ts,
            "group_id": group_id,
            "date_label": date_label,
            "enable_prune": enable_prune,
        }
        self.calls.append(call)
        return {
            "scanned_states": 1,
            "scanned_topics": 2,
            "cooling_transitions": 1,
            "closed_transitions": 1,
            "persisted_slices": 1,
            "pruned_topics": 0,
            "pruned_states": 0,
        }


def _run(coro):
    return asyncio.run(coro)


def test_topic_lifecycle_sweep_service_run_once_calls_manager() -> None:
    manager = _StubTopicManager()
    service = TopicLifecycleSweepService(
        topic_segment_manager=manager,  # type: ignore[arg-type]
        enabled=True,
        sweep_interval_seconds=30,
    )

    summary = _run(service.run_once(now_ts=123456))

    assert summary["scanned_states"] == 1
    assert len(manager.calls) == 1
    assert manager.calls[0]["now_ts"] == 123456
    assert manager.calls[0]["enable_prune"] is True


def test_topic_lifecycle_sweep_service_background_loop_start_and_stop() -> None:
    manager = _StubTopicManager()
    service = TopicLifecycleSweepService(
        topic_segment_manager=manager,  # type: ignore[arg-type]
        enabled=True,
        sweep_interval_seconds=10,
    )

    async def _case():
        started = service.start()
        assert started is True
        await asyncio.sleep(0.05)
        await service.stop()

    _run(_case())
    assert len(manager.calls) >= 1


def test_topic_lifecycle_sweep_service_disabled() -> None:
    manager = _StubTopicManager()
    service = TopicLifecycleSweepService(
        topic_segment_manager=manager,  # type: ignore[arg-type]
        enabled=False,
        sweep_interval_seconds=10,
    )

    async def _case():
        started = service.start()
        assert started is True
        await asyncio.sleep(0.05)
        await service.stop()

    _run(_case())
    assert manager.calls == []
