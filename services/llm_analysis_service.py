from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .models import LLMAnalysisConfig, LLMSemanticResult, MemberDigest, MessageRecord

DEFAULT_ANALYSIS_PROMPT_TEMPLATE = (
    "你是一个群聊日报语义分析助手。请根据提供的群聊消息，输出严格 JSON。\\n"
    "统计日期：{date_label}\\n"
    "统计范围：{time_window}\\n"
    "群组：{group_id}\\n"
    "消息样本（JSON 数组）：\\n{messages_json}\\n\\n"
    "要求：\\n"
    "1. 输出必须是 JSON 对象，不要输出额外解释。\\n"
    "2. 你需要给出热门话题、成员兴趣摘要、整体总结、建议 Bot 主动发言。\\n"
    "3. 热门话题控制在 3-{max_topics} 条。\\n"
    "4. 成员兴趣摘要的 key 请使用成员昵称。\\n"
    "5. suggested_bot_reply 要自然、友好、可直接发到群里，长度 1-2 句。\\n\\n"
    "JSON Schema（字段名必须一致）：\\n"
    "{{\\n"
    "  \"group_topics\": [\"话题1\", \"话题2\"],\\n"
    "  \"member_interests\": {{\"成员A\": \"兴趣摘要\", \"成员B\": \"兴趣摘要\"}},\\n"
    "  \"overall_summary\": \"整体总结\",\\n"
    "  \"suggested_bot_reply\": \"建议 Bot 主动发言\"\\n"
    "}}"
)


@dataclass
class LLMAnalysisOutcome:
    semantic: LLMSemanticResult | None = None
    notice: str = ""
    error: str = ""
    provider_id: str = ""
    provider_source: str = ""


class LLMAnalysisService:
    """封装 AstrBot 统一 LLM 调用能力，并提供结构化解析。"""

    async def analyze(
        self,
        *,
        context: Any,
        event: Any,
        config: LLMAnalysisConfig,
        group_id: str,
        date_label: str,
        time_window: str,
        messages: list[MessageRecord],
        active_members: list[MemberDigest],
        max_topics: int,
        resolved_provider_id: str | None = None,
        resolved_provider_source: str = "",
    ) -> LLMAnalysisOutcome:
        if not config.use_llm_topic_analysis:
            return LLMAnalysisOutcome(notice="语义分析已关闭（use_llm_topic_analysis=false）。")

        if resolved_provider_id is not None:
            provider_id = str(resolved_provider_id).strip()
            if config.analysis_provider_id.strip():
                source = "configured"
            else:
                source = resolved_provider_source or "session"
            provider_err = "" if provider_id else "未找到可用的分析模型 provider。"
        else:
            provider_id, source, provider_err = await self._resolve_provider_id(
                context=context,
                event=event,
                configured_provider_id=config.analysis_provider_id,
            )
        if not provider_id:
            return LLMAnalysisOutcome(error=provider_err or "未找到可用的分析模型 provider。")

        selected_messages = self._select_messages(messages, max_count=config.max_messages_for_analysis)
        messages_payload = self._build_messages_payload(selected_messages)

        analysis_prompt = self._build_analysis_prompt(
            config=config,
            group_id=group_id,
            date_label=date_label,
            time_window=time_window,
            messages_payload=messages_payload,
            active_members=active_members,
            max_topics=max_topics,
        )

        try:
            analysis_text = await self._llm_generate(
                context=context,
                provider_id=provider_id,
                prompt=analysis_prompt,
            )
            analysis_obj = self._parse_json_object(analysis_text)
            parsed = self._parse_unified_object(analysis_obj)
        except Exception as exc:
            return LLMAnalysisOutcome(
                error=f"语义分析模型调用或解析失败: {exc}",
                provider_id=provider_id,
                provider_source=source,
            )

        semantic = LLMSemanticResult(
            group_topics=parsed["group_topics"],
            member_interests=parsed["member_interests"],
            overall_summary=parsed["overall_summary"],
            suggested_bot_reply=parsed["suggested_bot_reply"],
        )
        return LLMAnalysisOutcome(
            semantic=semantic,
            notice=f"语义分析模型：{provider_id}（{source}）",
            provider_id=provider_id,
            provider_source=source,
        )

    async def resolve_provider_id(
        self,
        *,
        context: Any,
        event: Any,
        configured_provider_id: str,
    ) -> tuple[str, str, str]:
        """公开 provider 决策，供缓存命中校验复用。"""
        return await self._resolve_provider_id(
            context=context,
            event=event,
            configured_provider_id=configured_provider_id,
        )

    async def _resolve_provider_id(
        self,
        *,
        context: Any,
        event: Any,
        configured_provider_id: str,
    ) -> tuple[str, str, str]:
        configured_provider_id = configured_provider_id.strip()
        if configured_provider_id:
            return configured_provider_id, "configured", ""

        getter = getattr(context, "get_current_chat_provider_id", None)
        if callable(getter):
            umo = self._extract_unified_msg_origin(event)
            if not umo:
                return "", "", "无法从事件获取 unified_msg_origin，不能自动选择会话模型。"
            try:
                provider_id = await getter(umo=umo)
            except TypeError:
                provider_id = await getter(umo)
            except Exception as exc:
                return "", "", f"读取当前会话模型失败: {exc}"

            provider_id = str(provider_id).strip() if provider_id is not None else ""
            if provider_id:
                return provider_id, "session", ""
            return "", "", "当前会话没有可用模型，请配置 analysis_provider_id。"

        # TODO: 若未来 AstrBot 提供新的 provider 解析 API，可在此补充适配。
        return "", "", "当前 AstrBot 上下文未提供 get_current_chat_provider_id 接口。"

    def _extract_unified_msg_origin(self, event: Any) -> str:
        value = getattr(event, "unified_msg_origin", None)
        if value:
            return str(value)

        getter = getattr(event, "get_unified_msg_origin", None)
        if callable(getter):
            try:
                value = getter()
            except TypeError:
                try:
                    value = getter(event)  # pragma: no cover - 兼容部分实现可能要求显式传参
                except Exception:
                    value = None
            except Exception:
                value = None
            if value:
                return str(value)

        return ""

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
                if key in response and response[key]:
                    return str(response[key])

        text = getattr(response, "text", None)
        if text:
            return str(text)

        raise RuntimeError("LLM 返回中没有可解析的文本字段。")

    def _select_messages(self, messages: list[MessageRecord], max_count: int) -> list[MessageRecord]:
        ordered = sorted(messages, key=lambda item: item.timestamp)
        if max_count <= 0:
            return ordered
        return ordered[-max_count:]

    def _build_messages_payload(self, messages: list[MessageRecord]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for row in messages:
            payload.append(
                {
                    "timestamp": datetime.fromtimestamp(row.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_name": row.sender_name,
                    "sender_id": row.sender_id,
                    "content": row.content,
                }
            )
        return payload

    def _build_analysis_prompt(
        self,
        *,
        config: LLMAnalysisConfig,
        group_id: str,
        date_label: str,
        time_window: str,
        messages_payload: list[dict[str, Any]],
        active_members: list[MemberDigest],
        max_topics: int,
    ) -> str:
        template = config.analysis_prompt_template.strip() or DEFAULT_ANALYSIS_PROMPT_TEMPLATE
        active_member_names = [member.sender_name for member in active_members]
        prompt = template.format(
            group_id=group_id,
            date_label=date_label,
            time_window=time_window,
            max_topics=max_topics,
            message_count=len(messages_payload),
            messages_json=json.dumps(messages_payload, ensure_ascii=False, indent=2),
            active_member_names=", ".join(active_member_names),
        )
        interaction_style_hint = config.interaction_prompt_template.strip()
        if interaction_style_hint:
            prompt = (
                f"{prompt}\n\n"
                "补充风格要求（可选参考）：\n"
                f"{interaction_style_hint}\n\n"
                "注意：最终仍需输出同一个 JSON 对象，并包含 suggested_bot_reply 字段。"
            )
        return prompt

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("模型返回为空")

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])

        raise ValueError("模型返回不是合法 JSON 对象")

    def _parse_unified_object(self, data: dict[str, Any]) -> dict[str, Any]:
        topics_raw = data.get("group_topics", [])
        topics: list[str] = []
        if isinstance(topics_raw, list):
            for item in topics_raw:
                value = str(item).strip()
                if value:
                    topics.append(value)

        interests_raw = data.get("member_interests", {})
        member_interests: dict[str, str] = {}
        if isinstance(interests_raw, dict):
            for name, summary in interests_raw.items():
                key = str(name).strip()
                value = str(summary).strip()
                if key and value:
                    member_interests[key] = value

        overall_summary = str(data.get("overall_summary", "")).strip()

        suggested = str(data.get("suggested_bot_reply", "")).strip()
        if not suggested:
            interaction_obj = data.get("interaction", {})
            if isinstance(interaction_obj, dict):
                suggested = str(interaction_obj.get("suggested_bot_reply", "")).strip()
        if not suggested:
            suggested = str(data.get("bot_reply", "")).strip()

        if not topics:
            raise ValueError("group_topics 为空")
        if not overall_summary:
            raise ValueError("overall_summary 为空")
        if not suggested:
            raise ValueError("suggested_bot_reply 为空")

        return {
            "group_topics": topics,
            "member_interests": member_interests,
            "overall_summary": overall_summary,
            "suggested_bot_reply": suggested,
        }
