from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s


@dataclass(frozen=True)
class TextBM25Result:
    doc_id: str
    page: int
    score: float
    text: str
    source_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "score": self.score,
            "text": self.text,
            "source_path": self.source_path,
        }


class DocBenchBM25Retriever:
    """Load and query a DocBench BM25 index built from extracted page text only."""

    def __init__(self, index_dir: str | Path = "data/indexes/docbench_bm25"):
        self.index_dir = Path(index_dir)
        self.bm25_dir = self.index_dir / "bm25_index"
        if not self.bm25_dir.exists():
            raise FileNotFoundError(f"Missing BM25 index directory: {self.bm25_dir}")
        self.retriever = bm25s.BM25.load(self.bm25_dir, load_corpus=True)
        if not getattr(self.retriever, "corpus", None):
            raise ValueError(f"BM25 index has no corpus loaded: {self.bm25_dir}")
        self.corpus = list(self.retriever.corpus)

    def search(self, query: str, top_k: int = 5, doc_id: str | None = None) -> list[TextBM25Result]:
        if not query or not query.strip():
            return []
        query_tokens = bm25s.tokenize(query, show_progress=False)
        retrieve_k = len(self.corpus) if doc_id is not None else top_k
        results, scores = self.retriever.retrieve(
            query_tokens,
            k=retrieve_k,
            show_progress=False,
        )
        output: list[TextBM25Result] = []
        for record, score in zip(results[0], scores[0], strict=False):
            if doc_id is not None and str(record.get("doc_id")) != str(doc_id):
                continue
            output.append(
                TextBM25Result(
                    doc_id=str(record.get("doc_id", "")),
                    page=int(record.get("page", 0)),
                    score=float(score),
                    text=str(record.get("text", "")),
                    source_path=str(record.get("source_path", "")),
                )
            )
            if len(output) >= top_k:
                break
        return output
