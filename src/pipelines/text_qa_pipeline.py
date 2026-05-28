from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.retrieval.text_bm25_retriever import DocBenchBM25Retriever, TextBM25Result


@dataclass
class TextQAPipelineResult:
    answer: str
    retrieved_text_pages: list[dict[str, Any]]
    context: str
    prompt_profile: str | None
    prompt_name: str | None
    latency_total: float
    latency_retrieval: float
    latency_generation: float


class TextQAPipeline:
    """BM25 page retrieval followed by text-only generation."""

    def __init__(
        self,
        retriever: DocBenchBM25Retriever,
        generator: Any,
        top_k: int = 5,
        context_max_chars: int = 12000,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.context_max_chars = context_max_chars

    def build_context(self, pages: list[TextBM25Result]) -> str:
        chunks: list[str] = []
        total_chars = 0
        for page in pages:
            header = f"[doc_id={page.doc_id} page={page.page} score={page.score:.4f}]\n"
            body = page.text.strip()
            chunk = f"{header}{body}"
            remaining = self.context_max_chars - total_chars
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            total_chars += len(chunk)
        return "\n\n".join(chunks)

    def answer(
        self,
        question: str,
        doc_id: str | None = None,
        question_type: str = "text-only",
        context_mode: str = "text",
    ) -> TextQAPipelineResult:
        total_start = time.time()

        retrieval_start = time.time()
        pages = self.retriever.search(question, top_k=self.top_k, doc_id=doc_id)
        latency_retrieval = time.time() - retrieval_start

        context = self.build_context(pages)
        generation_start = time.time()
        if not context.strip():
            answer = "NOT FOUND"
        else:
            if hasattr(self.generator, "generate_answer_for_type"):
                answer = self.generator.generate_answer_for_type(
                    query=question,
                    question_type=question_type,
                    context_mode=context_mode,
                    context_images=None,
                    context_text=context,
                )
            else:
                answer = self.generator.generate_answer(
                    query=question,
                    context_images=None,
                    context_text=context,
                )
            if not answer or not str(answer).strip():
                answer = "NOT FOUND"
        latency_generation = time.time() - generation_start

        return TextQAPipelineResult(
            answer=str(answer).strip(),
            retrieved_text_pages=[page.to_json() for page in pages],
            context=context,
            prompt_profile=getattr(self.generator, "last_prompt_profile", None),
            prompt_name=getattr(self.generator, "last_prompt_name", None),
            latency_total=time.time() - total_start,
            latency_retrieval=latency_retrieval,
            latency_generation=latency_generation,
        )
