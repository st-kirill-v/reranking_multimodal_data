from __future__ import annotations

from pathlib import Path

from src.mmrag.config import PipelineConfig
from src.mmrag.dataset import parse_page_number
from src.mmrag.diagnostics import summarize_scores
from src.mmrag.embeddings import Qwen3PageEmbedder
from src.mmrag.indexing import FaissPageIndex
from src.mmrag.schema import RetrievalCandidate


class MultimodalPageRetriever:
    def __init__(self, config: PipelineConfig, *, strict_manifest: bool = True):
        self.config = config
        self.embedder = Qwen3PageEmbedder(config.embedder)
        self.page_index = FaissPageIndex(config).load()
        self._validate_embedding_contract(strict_manifest=strict_manifest)

    def _validate_embedding_contract(self, *, strict_manifest: bool) -> None:
        manifest_embedding = self.page_index.manifest.get("embedding")
        if not manifest_embedding:
            if strict_manifest:
                raise ValueError(
                    f"Index {self.page_index.index_path} has no manifest. Rebuild it with "
                    "scripts/build_page_index.py to avoid query/document encoder mismatch."
                )
            return
        expected = self.embedder.manifest(self.page_index.index.d)
        comparable_keys = [
            "model_id",
            "backend",
            "encoding_api",
            "normalize",
            "query_prompt",
            "dim",
        ]
        mismatches = {
            key: (manifest_embedding.get(key), expected.get(key))
            for key in comparable_keys
            if manifest_embedding.get(key) != expected.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Embedding contract mismatch for {self.page_index.index_path}: {mismatches}"
            )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalCandidate]:
        embedding = self.embedder.encode_query(query)
        return self.page_index.search(embedding, top_k or self.config.retrieval.first_stage_top_k)

    def search_with_debug(self, query: str, top_k: int | None = None) -> dict:
        candidates = self.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [candidate.to_json() for candidate in candidates],
            "score_summary": summarize_scores(candidates),
        }


def expand_with_neighbors(
    candidates: list[RetrievalCandidate], radius: int, final_limit: int | None = None
) -> list[RetrievalCandidate]:
    if radius <= 0:
        return candidates[:final_limit] if final_limit else candidates

    expanded: list[RetrievalCandidate] = []
    seen: set[str] = set()

    for candidate in candidates:
        for delta in range(-radius, radius + 1):
            page = candidate.page + delta
            if page < 1:
                continue
            path = Path(candidate.path).parent / f"page_{page}.png"
            if not path.exists() or parse_page_number(path) is None:
                continue
            key = f"{candidate.folder}_{page}"
            if key in seen:
                continue
            seen.add(key)
            score = candidate.score if delta == 0 else candidate.score * 0.8
            neighbor = RetrievalCandidate(
                folder=candidate.folder,
                page=page,
                path=path,
                score=score,
                rank=candidate.rank,
                index=candidate.index,
                source="neighbor_expansion" if delta != 0 else candidate.source,
                rerank_score=candidate.rerank_score if delta == 0 else None,
            )
            expanded.append(neighbor)

    expanded.sort(
        key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
        reverse=True,
    )
    return expanded[:final_limit] if final_limit else expanded
