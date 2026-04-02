from __future__ import annotations

import asyncio
import json
import urllib.request

from astrbot.api import logger


class APIEmbeddingBackend:
    """最小 API Embedding 后端（OpenAI-compatible /embeddings）。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        timeout_seconds: int = 10,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        self.timeout_seconds = max(3, int(timeout_seconds))

    async def embed_text(self, text: str) -> list[float] | None:
        if not self.api_key or not self.model:
            return None

        payload_text = str(text or "").strip()
        if not payload_text:
            return None

        try:
            return await asyncio.to_thread(self._embed_text_sync, payload_text)
        except Exception as exc:
            logger.warning(
                "[group_digest.embedding] api_embed_failed model=%s error=%s",
                self.model,
                exc,
            )
            return None

    def _embed_text_sync(self, text: str) -> list[float] | None:
        endpoint = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": text,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
            text_body = resp.read().decode("utf-8")
            data = json.loads(text_body)
        if not isinstance(data, dict):
            return None
        rows = data.get("data", [])
        if not isinstance(rows, list) or not rows:
            return None
        first = rows[0]
        if not isinstance(first, dict):
            return None
        embedding = first.get("embedding", [])
        if not isinstance(embedding, list):
            return None
        result: list[float] = []
        for item in embedding:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                return None
        return result or None

