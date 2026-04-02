from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from astrbot.api import logger

from .embedding.base import EmbeddingBackend
from .embedding.noop_backend import NoopEmbeddingBackend
from .embedding_store.base import EmbeddingStore
from .embedding_store.noop_store import NoopEmbeddingStore
from .llm_analysis_service import LLMAnalysisService
from .models import SlangExplanationRecord
from .slang_candidate_miner import SlangCandidate
from .slang_store import SlangStore


class SlangInterpretationService:
    """基于 RAG 语境的黑话解释服务（轻量版）。"""

    def __init__(
        self,
        *,
        llm_analysis_service: LLMAnalysisService | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        embedding_store: EmbeddingStore | None = None,
        slang_store: SlangStore | None = None,
        enable_slang_learning: bool = True,
        slang_retrieval_recent_days: int = 7,
        slang_retrieval_limit: int = 6,
        slang_min_context_items_for_inference: int = 2,
        slang_max_inference_per_build: int = 3,
        slang_reinfer_min_evidence_increase: int = 2,
    ) -> None:
        self.llm_analysis_service = llm_analysis_service or LLMAnalysisService()
        self.embedding_backend = embedding_backend or NoopEmbeddingBackend()
        self.embedding_store = embedding_store or NoopEmbeddingStore()
        self.slang_store = slang_store
        self.enable_slang_learning = bool(enable_slang_learning)
        self.slang_retrieval_recent_days = max(1, int(slang_retrieval_recent_days))
        self.slang_retrieval_limit = max(1, int(slang_retrieval_limit))
        self.slang_min_context_items_for_inference = max(
            1,
            int(slang_min_context_items_for_inference),
        )
        self.slang_max_inference_per_build = max(1, int(slang_max_inference_per_build))
        self.slang_reinfer_min_evidence_increase = max(
            1,
            int(slang_reinfer_min_evidence_increase),
        )

    async def resolve_candidates(
        self,
        *,
        context: Any,
        event: Any,
        analysis_provider_id: str,
        group_id: str,
        date_label: str,
        candidates: list[SlangCandidate],
    ) -> tuple[list[SlangExplanationRecord], dict[str, Any]]:
        if not self.enable_slang_learning:
            return [], {
                "enabled": False,
                "degraded": False,
                "inferred_count": 0,
                "reused_count": 0,
                "insufficient_context_count": 0,
            }
        if self.slang_store is None:
            return [], {
                "enabled": True,
                "degraded": True,
                "inferred_count": 0,
                "reused_count": 0,
                "insufficient_context_count": 0,
                "reason": "slang_store_missing",
            }
        if not candidates:
            return [], {
                "enabled": True,
                "degraded": False,
                "inferred_count": 0,
                "reused_count": 0,
                "insufficient_context_count": 0,
            }

        records: list[SlangExplanationRecord] = []
        inferred_count = 0
        reused_count = 0
        insufficient_context_count = 0
        degraded = False

        provider_id = ""
        provider_source = ""
        provider_err = ""

        for candidate in candidates:
            existing = self.slang_store.get(group_id=group_id, slang_term=candidate.term)
            if existing is not None and not self._should_reinfer(existing=existing, candidate=candidate):
                records.append(existing)
                reused_count += 1
                continue

            if inferred_count >= self.slang_max_inference_per_build:
                if existing is not None:
                    records.append(existing)
                    reused_count += 1
                continue

            rag_contexts, source_slice_ids = await self._retrieve_candidate_contexts(
                group_id=group_id,
                date_label=date_label,
                candidate=candidate,
            )
            if len(rag_contexts) < self.slang_min_context_items_for_inference:
                insufficient_context_count += 1
                if existing is not None:
                    records.append(existing)
                    reused_count += 1
                continue

            if not provider_id and not provider_err:
                provider_id, provider_source, provider_err = await self.llm_analysis_service.resolve_provider_id(
                    context=context,
                    event=event,
                    configured_provider_id=analysis_provider_id,
                )

            if not provider_id:
                degraded = True
                if existing is not None:
                    records.append(existing)
                    reused_count += 1
                continue

            inferred = await self._infer_record_with_llm(
                context=context,
                provider_id=provider_id,
                provider_source=provider_source,
                group_id=group_id,
                candidate=candidate,
                rag_contexts=rag_contexts,
                source_slice_ids=source_slice_ids,
            )
            if inferred is None:
                if existing is not None:
                    records.append(existing)
                    reused_count += 1
                continue

            self.slang_store.upsert(inferred)
            records.append(inferred)
            inferred_count += 1

        dedup: dict[str, SlangExplanationRecord] = {}
        for row in records:
            term = str(row.slang_term or "").strip()
            if not term:
                continue
            prev = dedup.get(term)
            if prev is None or int(row.updated_at or 0) >= int(prev.updated_at or 0):
                dedup[term] = row

        return list(dedup.values()), {
            "enabled": True,
            "degraded": degraded,
            "inferred_count": inferred_count,
            "reused_count": reused_count,
            "insufficient_context_count": insufficient_context_count,
            "provider_id": provider_id,
            "provider_error": provider_err,
        }

    def _should_reinfer(
        self,
        *,
        existing: SlangExplanationRecord,
        candidate: SlangCandidate,
    ) -> bool:
        previous_evidence = max(0, int(existing.evidence_count or 0))
        current_evidence = max(0, int(candidate.evidence_count or 0))
        if current_evidence <= previous_evidence:
            return False
        return (current_evidence - previous_evidence) >= self.slang_reinfer_min_evidence_increase

    async def _retrieve_candidate_contexts(
        self,
        *,
        group_id: str,
        date_label: str,
        candidate: SlangCandidate,
    ) -> tuple[list[str], list[str]]:
        if not self.embedding_store.enabled:
            return [], []

        query_text = self._build_candidate_query_text(candidate=candidate)
        if not query_text:
            return [], []

        try:
            vector = await self.embedding_backend.embed_text(query_text)
        except Exception as exc:
            logger.warning(
                "[group_digest.slang] candidate_embed_failed group_id=%s term=%s error=%s",
                group_id,
                candidate.term,
                exc,
            )
            return [], []
        if not vector:
            return [], []

        day_start_ts = self._resolve_day_start_ts(date_label=date_label)
        start_ts = day_start_ts - self.slang_retrieval_recent_days * 24 * 60 * 60
        end_ts = day_start_ts + 24 * 60 * 60
        try:
            rows = await self.embedding_store.query_topic_slices(
                group_id=group_id,
                query_vector=[float(item) for item in vector],
                start_ts=start_ts,
                end_ts=end_ts,
                recent_days=self.slang_retrieval_recent_days,
                limit=self.slang_retrieval_limit,
            )
        except Exception as exc:
            logger.warning(
                "[group_digest.slang] candidate_retrieval_failed group_id=%s term=%s error=%s",
                group_id,
                candidate.term,
                exc,
            )
            return [], []

        contexts: list[str] = []
        source_slice_ids: list[str] = []
        seen_slice_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            topic_id = str(row.get("topic_id", "")).strip()
            row_date = str(row.get("date_label", "")).strip()
            core_text = str(row.get("core_text", "")).strip()
            if not topic_id or not core_text:
                continue
            if len(core_text) > 180:
                core_text = f"{core_text[:180]}..."
            sid = f"{row_date}:{topic_id}" if row_date else topic_id
            if sid not in seen_slice_ids:
                source_slice_ids.append(sid)
                seen_slice_ids.add(sid)
            contexts.append(core_text)

        logger.info(
            "[group_digest.slang] candidate_retrieval group_id=%s term=%s query_chars=%d context_count=%d",
            group_id,
            candidate.term,
            len(query_text),
            len(contexts),
        )
        return contexts, source_slice_ids

    def _build_candidate_query_text(self, *, candidate: SlangCandidate) -> str:
        rows = [candidate.term]
        rows.extend(item for item in candidate.context_examples if item)
        return "\n".join(rows).strip()

    async def _infer_record_with_llm(
        self,
        *,
        context: Any,
        provider_id: str,
        provider_source: str,
        group_id: str,
        candidate: SlangCandidate,
        rag_contexts: list[str],
        source_slice_ids: list[str],
    ) -> SlangExplanationRecord | None:
        prompt = self._build_inference_prompt(
            group_id=group_id,
            candidate=candidate,
            rag_contexts=rag_contexts,
        )
        try:
            response_text = await self._llm_generate(
                context=context,
                provider_id=provider_id,
                prompt=prompt,
            )
            parsed = self._parse_json_object(response_text)
        except Exception as exc:
            logger.warning(
                "[group_digest.slang] inference_failed group_id=%s term=%s provider=%s(%s) error=%s",
                group_id,
                candidate.term,
                provider_id,
                provider_source or "-",
                exc,
            )
            return None

        if bool(parsed.get("no_info", False)):
            return None
        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            return None
        usage_context = str(parsed.get("usage_context", "")).strip()
        confidence = self._safe_float(parsed.get("confidence", 0.5), default=0.5)
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        now_ts = int(datetime.now().timestamp())
        evidence_from_llm = max(0, self._safe_int(parsed.get("evidence_count", 0), default=0))
        evidence_count = max(
            int(candidate.evidence_count),
            len(source_slice_ids),
            evidence_from_llm,
        )
        return SlangExplanationRecord(
            group_id=group_id,
            slang_term=candidate.term,
            explanation=explanation,
            usage_context=usage_context,
            confidence=confidence,
            evidence_count=evidence_count,
            source_slice_ids=list(source_slice_ids),
            source_semantic_unit_ids=[],
            created_at=now_ts,
            updated_at=now_ts,
        )

    def _build_inference_prompt(
        self,
        *,
        group_id: str,
        candidate: SlangCandidate,
        rag_contexts: list[str],
    ) -> str:
        contexts = "\n".join(
            f"{idx + 1}. {text}"
            for idx, text in enumerate(rag_contexts[: self.slang_retrieval_limit])
        )
        return (
            "你是群聊黑话解释助手。请基于候选词和历史语境，给出谨慎解释。\n"
            f"群组: {group_id}\n"
            f"候选黑话: {candidate.term}\n"
            f"候选统计信息: frequency={candidate.frequency}, slice_coverage={candidate.slice_coverage}\n"
            "历史语境片段:\n"
            f"{contexts}\n\n"
            "输出严格 JSON，不要输出额外说明：\n"
            "{\n"
            "  \"slang_term\": \"候选词\",\n"
            "  \"explanation\": \"对黑话含义的简洁解释\",\n"
            "  \"usage_context\": \"典型使用场景\",\n"
            "  \"confidence\": 0.0,\n"
            "  \"evidence_count\": 0,\n"
            "  \"no_info\": false\n"
            "}\n"
            "如果证据不足，请设置 no_info=true，并把 explanation 留空。"
        )

    async def _llm_generate(self, *, context: Any, provider_id: str, prompt: str) -> str:
        llm_generate = getattr(context, "llm_generate", None)
        if not callable(llm_generate):
            raise RuntimeError("当前 AstrBot 上下文未提供 llm_generate 接口。")
        response = await llm_generate(chat_provider_id=provider_id, prompt=prompt)
        text = getattr(response, "completion_text", None)
        if text:
            return str(text)
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("completion_text", "text", "content"):
                value = response.get(key)
                if value:
                    return str(value)
        alt = getattr(response, "text", None)
        if alt:
            return str(alt)
        raise RuntimeError("LLM 返回中没有可解析文本。")

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        normalized = str(text or "").strip()
        if not normalized:
            raise RuntimeError("empty_response")
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", normalized, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            normalized = fence_match.group(1).strip()

        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            fragment = normalized[start : end + 1]
            parsed = json.loads(fragment)
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("json_parse_failed")

    def _resolve_day_start_ts(self, *, date_label: str) -> int:
        try:
            day = datetime.strptime(date_label, "%Y-%m-%d")
            return int(day.timestamp())
        except Exception:
            now = datetime.now()
            fallback = datetime(now.year, now.month, now.day)
            return int(fallback.timestamp())

    def _safe_int(self, value: object, *, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def _safe_float(self, value: object, *, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

