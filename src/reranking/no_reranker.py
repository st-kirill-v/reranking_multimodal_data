from __future__ import annotations

from typing import Any


class NoReranker:
    name = "no_reranking"

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(candidates)
