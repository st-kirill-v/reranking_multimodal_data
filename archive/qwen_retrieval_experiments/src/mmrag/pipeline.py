from __future__ import annotations

from PIL import Image

from src.core.generators.qwen_vl_generator import create_table_generator
from src.mmrag.config import PipelineConfig
from src.mmrag.rerank import NemotronVLReranker
from src.mmrag.retrieval import MultimodalPageRetriever, expand_with_neighbors
from src.mmrag.schema import RetrievalCandidate


class PageRAGPipeline:
    def __init__(
        self, config: PipelineConfig, *, use_reranker: bool = True, use_generator: bool = True
    ):
        self.config = config
        self.retriever = MultimodalPageRetriever(config)
        self.reranker = NemotronVLReranker(config.reranker) if use_reranker else None
        self.generator = (
            create_table_generator(
                device=config.generator.device,
                max_image_long_edge=config.generator.max_image_long_edge,
            )
            if use_generator
            else None
        )

    def retrieve(self, query: str) -> list[RetrievalCandidate]:
        candidates = self.retriever.search(query, top_k=self.config.retrieval.first_stage_top_k)
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)[
                : self.config.retrieval.rerank_top_k
            ]
        return expand_with_neighbors(
            candidates,
            radius=self.config.retrieval.neighbor_radius,
            final_limit=self.config.retrieval.final_top_k,
        )

    def answer(self, query: str) -> dict:
        candidates = self.retrieve(query)
        images: list[Image.Image] = []
        for candidate in candidates[: self.config.generator.max_pages]:
            with Image.open(candidate.path) as img:
                images.append(img.convert("RGB").copy())
        if not images or self.generator is None:
            return {
                "answer": "NOT FOUND",
                "pages": [candidate.to_json() for candidate in candidates],
            }
        answer = self.generator.generate_answer(query, images)
        return {
            "answer": answer,
            "pages": [candidate.to_json() for candidate in candidates],
        }
