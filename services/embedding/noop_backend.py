from __future__ import annotations


class NoopEmbeddingBackend:
    async def embed_text(self, text: str) -> list[float] | None:
        _ = text
        return None

