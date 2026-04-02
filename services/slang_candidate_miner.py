from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .models import TopicSliceRecord


_DEFAULT_STOPWORDS = frozenset(
    {
        "今天",
        "昨天",
        "明天",
        "现在",
        "然后",
        "因为",
        "所以",
        "感觉",
        "觉得",
        "可以",
        "应该",
        "需要",
        "已经",
        "真的",
        "确实",
        "我们",
        "你们",
        "他们",
        "这个",
        "那个",
        "事情",
        "问题",
        "消息",
        "讨论",
        "安排",
        "上线",
        "回滚",
        "版本",
        "方案",
        "测试",
        "部署",
        "复盘",
        "哈哈",
        "好的",
        "收到",
        "嗯嗯",
        "可以",
        "就是",
        "不是",
        "没有",
        "一下",
    }
)


@dataclass(frozen=True)
class SlangCandidate:
    term: str
    score: float
    frequency: int
    slice_coverage: int
    evidence_count: int
    source_slice_ids: list[str] = field(default_factory=list)
    context_examples: list[str] = field(default_factory=list)


class SlangCandidateMiner:
    """黑话候选统计预筛（低成本）。"""

    TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,15}|[\u4e00-\u9fff]{2,8}")

    def __init__(
        self,
        *,
        min_term_frequency: int = 2,
        min_slice_coverage: int = 2,
        max_candidates: int = 10,
        min_term_length: int = 2,
        max_term_length: int = 12,
        current_day_boost: float = 0.4,
    ) -> None:
        self.min_term_frequency = max(1, int(min_term_frequency))
        self.min_slice_coverage = max(1, int(min_slice_coverage))
        self.max_candidates = max(1, int(max_candidates))
        self.min_term_length = max(1, int(min_term_length))
        self.max_term_length = max(self.min_term_length, int(max_term_length))
        self.current_day_boost = float(current_day_boost)

    def mine_candidates(
        self,
        *,
        current_day_slices: list[TopicSliceRecord],
        recent_slices: list[TopicSliceRecord],
        exclude_terms: set[str] | None = None,
    ) -> list[SlangCandidate]:
        exclude = {str(item).strip() for item in (exclude_terms or set()) if str(item).strip()}
        merged = self._merge_slice_rows(current_day_slices=current_day_slices, recent_slices=recent_slices)
        if not merged:
            return []

        current_day_slice_ids = {self._slice_key(row) for row in current_day_slices}
        term_frequency: dict[str, int] = {}
        term_slice_ids: dict[str, set[str]] = {}
        term_contexts: dict[str, list[str]] = {}

        for row in merged:
            slice_key = self._slice_key(row)
            text_blocks = [str(row.core_text or "").strip()]
            text_blocks.extend(str(item).strip() for item in row.recent_keywords if str(item).strip())
            combined = "\n".join(item for item in text_blocks if item).strip()
            if not combined:
                continue

            tokens = self._extract_tokens(combined)
            if not tokens:
                continue
            for term in tokens:
                if term in exclude:
                    continue
                term_frequency[term] = term_frequency.get(term, 0) + 1
                term_slice_ids.setdefault(term, set()).add(slice_key)
                bucket = term_contexts.setdefault(term, [])
                if len(bucket) < 4:
                    bucket.append(combined[:200])

        candidates: list[SlangCandidate] = []
        for term, frequency in term_frequency.items():
            if frequency < self.min_term_frequency:
                continue
            coverage = len(term_slice_ids.get(term, set()))
            if coverage < self.min_slice_coverage:
                continue

            source_ids = sorted(term_slice_ids.get(term, set()))
            current_hits = sum(1 for item in source_ids if item in current_day_slice_ids)
            score = float(frequency) + float(coverage) * 1.2 + float(current_hits) * self.current_day_boost
            candidates.append(
                SlangCandidate(
                    term=term,
                    score=round(score, 4),
                    frequency=frequency,
                    slice_coverage=coverage,
                    evidence_count=coverage,
                    source_slice_ids=source_ids,
                    context_examples=list(term_contexts.get(term, []))[:3],
                )
            )

        candidates.sort(key=lambda item: (item.score, item.frequency, item.slice_coverage), reverse=True)
        return candidates[: self.max_candidates]

    def _extract_tokens(self, text: str) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for match in self.TOKEN_PATTERN.finditer(str(text or "")):
            token = str(match.group(0) or "").strip()
            if not token:
                continue
            for candidate in self._expand_token_candidates(token):
                if candidate in seen:
                    continue
                seen.add(candidate)
                rows.append(candidate)
        return rows

    def _expand_token_candidates(self, token: str) -> list[str]:
        base = str(token or "").strip()
        if not base:
            return []

        result: list[str] = []
        if self._is_valid_token(base):
            result.append(base)

        # 对连续中文长串补充轻量 n-gram 子片段，降低“整段命中却漏掉核心词”的概率。
        if self._is_cjk_token(base) and len(base) >= 4:
            for size in (2, 3, 4):
                if size > len(base):
                    continue
                for idx in range(0, len(base) - size + 1):
                    sub = base[idx : idx + size]
                    if self._is_valid_token(sub):
                        result.append(sub)
        return result

    def _is_valid_token(self, token: str) -> bool:
        if len(token) < self.min_term_length or len(token) > self.max_term_length:
            return False
        if token.lower() in _DEFAULT_STOPWORDS:
            return False
        if token in _DEFAULT_STOPWORDS:
            return False
        if token.isdigit():
            return False
        return True

    def _is_cjk_token(self, token: str) -> bool:
        return all("\u4e00" <= ch <= "\u9fff" for ch in token)

    def _merge_slice_rows(
        self,
        *,
        current_day_slices: list[TopicSliceRecord],
        recent_slices: list[TopicSliceRecord],
    ) -> list[TopicSliceRecord]:
        mapping: dict[str, TopicSliceRecord] = {}
        for row in self._iter_rows(current_day_slices, recent_slices):
            key = self._slice_key(row)
            mapping[key] = row
        return list(mapping.values())

    def _iter_rows(self, *groups: Iterable[TopicSliceRecord]) -> Iterable[TopicSliceRecord]:
        for group in groups:
            for row in group:
                if isinstance(row, TopicSliceRecord):
                    yield row

    def _slice_key(self, row: TopicSliceRecord) -> str:
        return f"{row.date_label}:{row.topic_id}"
