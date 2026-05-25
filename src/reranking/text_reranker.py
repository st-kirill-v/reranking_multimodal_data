from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


class TextReranker:
    """Lightweight text/OCR/metadata reranker for ablation.

    It uses candidate text-like fields when available and otherwise falls back to the input order.
    """

    name = "text_reranking"

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_tokens = _tokens(question)
        ranked = []
        for idx, candidate in enumerate(candidates):
            text = " ".join(
                str(candidate.get(key, ""))
                for key in ("text", "ocr_text", "caption", "title", "folder", "page")
            )
            c_tokens = _tokens(text)
            score = len(q_tokens & c_tokens) / max(len(q_tokens | c_tokens), 1)
            item = dict(candidate)
            item["text_rerank_score"] = score
            item["text_rerank_rank"] = idx + 1
            ranked.append((score, -idx, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _score, _idx, item in ranked]
