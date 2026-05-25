from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
