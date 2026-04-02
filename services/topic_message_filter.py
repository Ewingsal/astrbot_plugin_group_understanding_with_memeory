from __future__ import annotations

import re
from dataclasses import dataclass


AGREEMENT_PHRASES = {
    "对",
    "是啊",
    "我也觉得",
    "确实",
    "同意",
    "没错",
    "有道理",
}

TONE_PHRASES = {
    "哈哈哈",
    "哈哈",
    "hhh",
    "hh",
    "笑死",
    "啊这",
    "额",
    "呃",
}

ACK_PHRASES = {
    "嗯嗯",
    "好的",
    "收到",
    "ok",
    "okk",
    "明白",
}

ACTION_HINTS = {
    "做",
    "改",
    "修",
    "发",
    "看",
    "写",
    "提",
    "开",
    "复盘",
    "安排",
    "讨论",
    "上线",
    "发布",
    "训练",
    "吃",
    "买",
    "跑",
    "打",
}

TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2})|(\d{1,2}点)|(\d{1,2}号)|(\d{4}-\d{1,2}-\d{1,2})")
ENTITY_PATTERN = re.compile(r"@[A-Za-z0-9_\-\u4e00-\u9fff]+|#[A-Za-z0-9_\-\u4e00-\u9fff]+")
ALNUM_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{3,}")
HAN_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{3,}")
PUNCT_OR_SPACE_PATTERN = re.compile(r"[\s\u3000,，.。!！?？~～…·、]+")


@dataclass(frozen=True)
class TopicMessageFilterResult:
    is_effective: bool
    reason: str


def classify_topic_message(text: str) -> TopicMessageFilterResult:
    raw = str(text or "").strip()
    if not raw:
        return TopicMessageFilterResult(is_effective=False, reason="empty")

    normalized = _normalize_text(raw)
    if normalized in AGREEMENT_PHRASES:
        return TopicMessageFilterResult(is_effective=False, reason="pure_agreement")
    if normalized in TONE_PHRASES:
        return TopicMessageFilterResult(is_effective=False, reason="pure_tone")
    if normalized in ACK_PHRASES:
        return TopicMessageFilterResult(is_effective=False, reason="pure_ack")

    if _is_short_and_low_information(raw):
        return TopicMessageFilterResult(is_effective=False, reason="short_low_information")

    return TopicMessageFilterResult(is_effective=True, reason="effective")


def is_effective_topic_message(text: str) -> bool:
    return classify_topic_message(text).is_effective


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = PUNCT_OR_SPACE_PATTERN.sub("", lowered)
    return lowered


def _is_short_and_low_information(text: str) -> bool:
    raw = str(text or "").strip()
    compact = PUNCT_OR_SPACE_PATTERN.sub("", raw)
    if len(compact) > 6:
        return False

    if TIME_PATTERN.search(raw):
        return False
    if ENTITY_PATTERN.search(raw):
        return False
    if ALNUM_TOKEN_PATTERN.search(raw):
        return False

    lowered = compact.lower()
    for action in ACTION_HINTS:
        if action in lowered:
            return False

    # 极短句里若包含较明确中文词块，放行为有效消息，避免误杀。
    if HAN_TOKEN_PATTERN.search(compact):
        return False

    return True
