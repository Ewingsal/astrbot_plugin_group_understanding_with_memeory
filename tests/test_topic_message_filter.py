from __future__ import annotations

from astrbot_plugin_group_digest.services.topic_message_filter import classify_topic_message


def test_filter_pure_agreement_message() -> None:
    result = classify_topic_message("我也觉得")
    assert result.is_effective is False
    assert result.reason == "pure_agreement"


def test_filter_pure_tone_message() -> None:
    result = classify_topic_message("哈哈哈")
    assert result.is_effective is False
    assert result.reason == "pure_tone"


def test_filter_pure_ack_message() -> None:
    result = classify_topic_message("收到")
    assert result.is_effective is False
    assert result.reason == "pure_ack"


def test_filter_short_low_information_message() -> None:
    result = classify_topic_message("好耶")
    assert result.is_effective is False
    assert result.reason == "short_low_information"


def test_keep_normal_effective_message() -> None:
    result = classify_topic_message("今晚8点我们讨论部署回滚方案")
    assert result.is_effective is True
    assert result.reason == "effective"

