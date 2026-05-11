from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

from src.mmrag.config import RerankerConfig
from src.mmrag.schema import RetrievalCandidate


class NemotronVLReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
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
        reranked: list[RetrievalCandidate] = []
        for offset in range(0, len(candidates), self.config.batch_size):
            batch = candidates[offset : offset + self.config.batch_size]
            examples = []
            valid: list[RetrievalCandidate] = []
            for candidate in batch:
                try:
                    with Image.open(candidate.path) as img:
                        page_image = img.convert("RGB").copy()
                except OSError:
                    continue
                examples.append({"question": query, "doc_text": "", "doc_image": page_image})
                valid.append(candidate)
            if not examples:
                continue

            batch_dict = self.processor.process_queries_documents_crossencoder(examples)
            batch_dict = {
                key: value.to(self.config.device)
                for key, value in batch_dict.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.no_grad():
                logits = self.model(**batch_dict, return_dict=True).logits
            for candidate, logit in zip(valid, logits):
                candidate.rerank_score = float(torch.sigmoid(logit).item())
                reranked.append(candidate)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reranked.sort(key=lambda item: item.rerank_score or float("-inf"), reverse=True)
        return reranked
