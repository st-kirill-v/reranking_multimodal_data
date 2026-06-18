from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any


TEXT_RERANKER_MODELS = {
    "bge-base": "BAAI/bge-reranker-base",
    "bge-large": "BAAI/bge-reranker-large",
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "jina": "jinaai/jina-reranker-v2-base-multilingual",
}

TEXT_RERANKER_MAX_LENGTH_CAPS = {
    "baai/bge-reranker-base": 512,
    "baai/bge-reranker-large": 512,
    "cross-encoder/ms-marco-minilm-l-6-v2": 512,
    "jinaai/jina-reranker-v2-base-multilingual": 512,
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


@dataclass
class TextRerankOutput:
    candidates: list[Any]
    latency: float
    model_id: str
    backend: str


class LexicalTextReranker:
    """Small fallback reranker used only for smoke/dry runs."""

    name = "lexical_text_reranker"

    def __init__(self, model_id: str = "lexical") -> None:
        self.model_id = model_id

    def rerank(self, question: str, candidates: list[Any]) -> TextRerankOutput:
        start = time.time()
        q_tokens = _tokens(question)
        ranked = []
        for idx, candidate in enumerate(candidates):
            text = getattr(candidate, "text", "")
            c_tokens = _tokens(text)
            score = len(q_tokens & c_tokens) / max(len(q_tokens | c_tokens), 1)
            item = _with_text_score(candidate, score, idx + 1)
            ranked.append((score, -idx, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return TextRerankOutput(
            candidates=[item for _score, _idx, item in ranked],
            latency=time.time() - start,
            model_id=self.model_id,
            backend=self.name,
        )


class CrossEncoderTextReranker:
    """Cross-encoder text reranker for DocBench page-text ablations."""

    name = "cross_encoder_text_reranker"

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 4096,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = TEXT_RERANKER_MODELS.get(model_id, model_id)
        self.device = device
        self.batch_size = batch_size
        self.max_length = self._resolve_max_length(max_length)
        self.trust_remote_code = trust_remote_code
        self.model = self._load_model()

    def _resolve_max_length(self, requested_max_length: int) -> int:
        cap = TEXT_RERANKER_MAX_LENGTH_CAPS.get(self.model_id.lower())
        if cap is not None and requested_max_length > cap:
            print(
                f"[TextReranker] max_length={requested_max_length} is too high for "
                f"{self.model_id}; using max_length={cap}."
            )
            return cap
        return requested_max_length

    def _load_model(self) -> Any:
        if "jina-reranker-v2" in self.model_id.lower():
            _patch_jina_transformers_compat()

        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {
            "model_name": self.model_id,
            "device": self.device,
            "max_length": self.max_length,
        }
        try:
            return CrossEncoder(
                **kwargs,
                trust_remote_code=self.trust_remote_code,
                automodel_args={"trust_remote_code": self.trust_remote_code},
                tokenizer_args={"trust_remote_code": self.trust_remote_code},
            )
        except TypeError:
            try:
                return CrossEncoder(
                    **kwargs,
                    automodel_args={"trust_remote_code": self.trust_remote_code},
                    tokenizer_args={"trust_remote_code": self.trust_remote_code},
                )
            except TypeError:
                return CrossEncoder(**kwargs)

    def rerank(self, question: str, candidates: list[Any]) -> TextRerankOutput:
        start = time.time()
        pairs = [(question, getattr(candidate, "text", "") or "") for candidate in candidates]
        if not pairs:
            return TextRerankOutput([], 0.0, self.model_id, self.name)
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=False)):
            item = _with_text_score(candidate, float(score), idx + 1)
            ranked.append((float(score), -idx, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        reranked = []
        for rank, (_score, _idx, item) in enumerate(ranked, start=1):
            reranked.append(_with_text_score(item, float(_score), rank))
        return TextRerankOutput(
            candidates=reranked,
            latency=time.time() - start,
            model_id=self.model_id,
            backend=self.name,
        )


def _with_text_score(candidate: Any, score: float, rank: int) -> Any:
    if hasattr(candidate, "text_rerank_score"):
        candidate.text_rerank_score = score
        candidate.text_rerank_rank = rank
        return candidate
    item = dict(candidate)
    item["text_rerank_score"] = score
    item["text_rerank_rank"] = rank
    return item


def _patch_jina_transformers_compat() -> None:
    """Patch a removed transformers helper expected by Jina's remote code.

    Some recent transformers versions no longer expose
    xlm_roberta.modeling_xlm_roberta.create_position_ids_from_input_ids, while
    jinaai/jina-reranker-v2-base-multilingual imports it from that location.
    The equivalent helper is still available in the RoBERTa module.
    """

    try:
        import torch
        from transformers.models.xlm_roberta import modeling_xlm_roberta
    except Exception:
        return

    def create_position_ids_from_input_ids(
        input_ids: Any,
        padding_idx: int,
        past_key_values_length: int = 0,
    ) -> Any:
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    if not hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
        modeling_xlm_roberta.create_position_ids_from_input_ids = create_position_ids_from_input_ids


def create_text_reranker(
    model_id: str,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 4096,
    trust_remote_code: bool = True,
    backend: str = "cross_encoder",
) -> CrossEncoderTextReranker | LexicalTextReranker:
    if backend == "lexical" or model_id == "lexical":
        return LexicalTextReranker(model_id=model_id)
    return CrossEncoderTextReranker(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )
