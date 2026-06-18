from __future__ import annotations

import os
import time

import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

from src.mmrag.config import RerankerConfig
from src.mmrag.schema import RetrievalCandidate


class NemotronVLReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.profile_enabled = os.getenv("MMRAG_PROFILE_RERANK", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self.last_profile: dict[str, float | int] = {}
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map=config.device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            max_input_tiles=config.max_input_tiles,
            use_thumbnail=config.use_thumbnail,
            rerank_max_length=config.max_length,
        )

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        total_start = time.perf_counter()
        profile = {
            "batches": 0,
            "candidates": len(candidates),
            "image_load_seconds": 0.0,
            "processor_seconds": 0.0,
            "to_device_seconds": 0.0,
            "forward_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "empty_cache_seconds": 0.0,
        }
        reranked: list[RetrievalCandidate] = []
        for offset in range(0, len(candidates), self.config.batch_size):
            profile["batches"] += 1
            batch = candidates[offset : offset + self.config.batch_size]
            examples = []
            valid: list[RetrievalCandidate] = []
            step_start = time.perf_counter()
            for candidate in batch:
                try:
                    with Image.open(candidate.path) as img:
                        page_image = img.convert("RGB").copy()
                except OSError:
                    continue
                examples.append({"question": query, "doc_text": "", "doc_image": page_image})
                valid.append(candidate)
            profile["image_load_seconds"] += time.perf_counter() - step_start
            if not examples:
                continue

            step_start = time.perf_counter()
            batch_dict = self.processor.process_queries_documents_crossencoder(examples)
            profile["processor_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            batch_dict = {
                key: value.to(self.config.device)
                for key, value in batch_dict.items()
                if isinstance(value, torch.Tensor)
            }
            profile["to_device_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            with torch.no_grad():
                logits = self.model(**batch_dict, return_dict=True).logits
            profile["forward_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            for candidate, logit in zip(valid, logits):
                candidate.rerank_score = float(torch.sigmoid(logit).item())
                reranked.append(candidate)
            profile["postprocess_seconds"] += time.perf_counter() - step_start

            if torch.cuda.is_available():
                step_start = time.perf_counter()
                torch.cuda.empty_cache()
                profile["empty_cache_seconds"] += time.perf_counter() - step_start

        reranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        profile["total_seconds"] = time.perf_counter() - total_start
        self.last_profile = profile
        if self.profile_enabled:
            print(
                "[Rerank profile] "
                f"total={profile['total_seconds']:.2f}s "
                f"batches={profile['batches']} candidates={profile['candidates']} "
                f"image_load={profile['image_load_seconds']:.2f}s "
                f"processor={profile['processor_seconds']:.2f}s "
                f"to_device={profile['to_device_seconds']:.2f}s "
                f"forward={profile['forward_seconds']:.2f}s "
                f"postprocess={profile['postprocess_seconds']:.2f}s "
                f"empty_cache={profile['empty_cache_seconds']:.2f}s"
            )
        return reranked


class NemotronVLTextImageReranker(NemotronVLReranker):
    """Nemotron VL reranker that sends page image plus extracted page text.

    This is intentionally separate from NemotronVLReranker so the old image-only
    reranking baseline remains unchanged.
    """

    def __init__(
        self,
        config: RerankerConfig,
        *,
        evidence_map: dict[tuple[str, int], dict[str, object]],
        max_text_chars: int = 4096,
    ) -> None:
        super().__init__(config)
        self.evidence_map = evidence_map
        self.max_text_chars = max_text_chars

    def _candidate_text(self, candidate: RetrievalCandidate) -> str:
        evidence = self.evidence_map.get((str(candidate.folder), int(candidate.page)), {})
        text = str(evidence.get("text") or "")
        if self.max_text_chars <= 0:
            return ""
        return text[: self.max_text_chars]

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        total_start = time.perf_counter()
        profile = {
            "batches": 0,
            "candidates": len(candidates),
            "image_load_seconds": 0.0,
            "text_chars": 0,
            "processor_seconds": 0.0,
            "to_device_seconds": 0.0,
            "forward_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "empty_cache_seconds": 0.0,
        }
        reranked: list[RetrievalCandidate] = []
        for offset in range(0, len(candidates), self.config.batch_size):
            profile["batches"] += 1
            batch = candidates[offset : offset + self.config.batch_size]
            examples = []
            valid: list[RetrievalCandidate] = []
            step_start = time.perf_counter()
            for candidate in batch:
                try:
                    with Image.open(candidate.path) as img:
                        page_image = img.convert("RGB").copy()
                except OSError:
                    continue
                doc_text = self._candidate_text(candidate)
                profile["text_chars"] += len(doc_text)
                candidate.rerank_text_chars = len(doc_text)  # type: ignore[attr-defined]
                examples.append({"question": query, "doc_text": doc_text, "doc_image": page_image})
                valid.append(candidate)
            profile["image_load_seconds"] += time.perf_counter() - step_start
            if not examples:
                continue

            step_start = time.perf_counter()
            batch_dict = self.processor.process_queries_documents_crossencoder(examples)
            profile["processor_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            batch_dict = {
                key: value.to(self.config.device)
                for key, value in batch_dict.items()
                if isinstance(value, torch.Tensor)
            }
            profile["to_device_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            with torch.no_grad():
                logits = self.model(**batch_dict, return_dict=True).logits
            profile["forward_seconds"] += time.perf_counter() - step_start
            step_start = time.perf_counter()
            for candidate, logit in zip(valid, logits):
                candidate.rerank_score = float(torch.sigmoid(logit).item())
                reranked.append(candidate)
            profile["postprocess_seconds"] += time.perf_counter() - step_start

            if torch.cuda.is_available():
                step_start = time.perf_counter()
                torch.cuda.empty_cache()
                profile["empty_cache_seconds"] += time.perf_counter() - step_start

        reranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        profile["total_seconds"] = time.perf_counter() - total_start
        self.last_profile = profile
        if self.profile_enabled:
            print(
                "[Rerank text+image profile] "
                f"total={profile['total_seconds']:.2f}s "
                f"batches={profile['batches']} candidates={profile['candidates']} "
                f"text_chars={profile['text_chars']} "
                f"image_load={profile['image_load_seconds']:.2f}s "
                f"processor={profile['processor_seconds']:.2f}s "
                f"to_device={profile['to_device_seconds']:.2f}s "
                f"forward={profile['forward_seconds']:.2f}s "
                f"postprocess={profile['postprocess_seconds']:.2f}s "
                f"empty_cache={profile['empty_cache_seconds']:.2f}s"
            )
        return reranked
