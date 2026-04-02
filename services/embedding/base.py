from __future__ import annotations

from typing import Protocol


class EmbeddingBackend(Protocol):
    async def embed_text(self, text: str) -> list[float] | None:
        """返回 embedding 向量；不可用时返回 None。"""

