"""
Мультимодальный поиск на основе JinaCLIP (быстрый) + Nemotron VL (точный).
"""

import torch
from transformers import AutoModel, AutoProcessor, AutoModelForSequenceClassification
from pathlib import Path
import faiss
import json
import numpy as np
from PIL import Image
import time


class MultimodalSearch:
    def __init__(self, models_dir: str, index_dir: str):
        self.models_dir = Path(models_dir)
        self.index_dir = Path(index_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        self._load_jina()
        self._load_nemotron()
        self._load_index()

    def _load_jina(self):
        """Загрузка JinaCLIP для быстрого первого прохода"""
        print("Loading JinaCLIP model for first-stage retrieval...")
        self.jina_model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v1", trust_remote_code=True
        ).to(self.device)
        self.jina_processor = AutoProcessor.from_pretrained(
            "jinaai/jina-clip-v1", trust_remote_code=True
        )
        print("JinaCLIP model loaded")

    def _load_nemotron(self):
        """Загрузка Nemotron Embed и Rerank"""
        embed_path = self.models_dir / "embed-vl-1b-v2"
        rerank_path = self.models_dir / "rerank-vl-1b-v2"

        self.embed_model = None
        print("Nemotron Embed model disabled (skipped)")

        print("Loading Nemotron Rerank model from HuggingFace cache...")
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map=self.device,
        ).eval()

        if rerank_path.exists():
            self.rerank_processor = AutoProcessor.from_pretrained(
                rerank_path,
                trust_remote_code=True,
                local_files_only=True,
                max_input_tiles=6,
                use_thumbnail=True,
                rerank_max_length=2048,
            )
        else:
            self.rerank_processor = AutoProcessor.from_pretrained(
                "nvidia/llama-nemotron-rerank-vl-1b-v2",
                trust_remote_code=True,
                max_input_tiles=6,
                use_thumbnail=True,
                rerank_max_length=2048,
            )
        print("Nemotron Rerank model loaded")

    def _load_index(self):
        """Загрузка FAISS индекса (JinaCLIP) и метаданных"""
        index_path = self.index_dir / "pages_jina.index"
        metadata_path = self.index_dir / "metadata_jina.json"

        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"JinaCLIP index loaded: {self.index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Index not found at {index_path}")

    def search(self, query: str, n_results: int = 10, use_rerank: bool = True) -> dict:
        """Двухэтапный поиск: JinaCLIP (быстро) → Nemotron (точно)"""
        start_time = time.time()

        candidates = self._jina_search(query, n_results * 5 if use_rerank else n_results)

        if use_rerank and self.rerank_model and candidates:
            candidates = self._nemotron_rerank(query, candidates, n_results)
        else:
            candidates = candidates[:n_results]

        search_time = time.time() - start_time

        return {
            "query": query,
            "results": candidates,
            "search_time": search_time,
            "total_found": len(candidates),
            "rerank_used": use_rerank and self.rerank_model is not None,
        }

    def _jina_search(self, query: str, n_results: int) -> list:
        """Первый этап: быстрый поиск через JinaCLIP"""
        with torch.no_grad():
            text_embedding = self.jina_model.encode_text([query])
            text_embedding = text_embedding / torch.norm(text_embedding, p=2, dim=-1, keepdim=True)

        scores, indices = self.index.search(text_embedding.cpu().numpy(), n_results)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                meta = self.metadata[idx]
                candidates.append(
                    {
                        "score": float(score),
                        "folder": meta["folder"],
                        "page": meta["page"],
                        "path": meta["path"],
                        "index": int(idx),
                    }
                )

        return candidates

    def _nemotron_rerank(self, query: str, candidates: list, n_results: int) -> list:
        """Финальный этап: реранкинг через Nemotron Rerank"""
        torch.cuda.empty_cache()

        reranked = []
        batch_size = 2

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]

            examples = []
            for cand in batch:
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    examples.append({"question": query, "doc_text": "", "doc_image": img})
                except:
                    continue

            if not examples:
                continue

            try:
                batch_dict = self.rerank_processor.process_queries_documents_crossencoder(examples)
                batch_dict = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_dict.items()
                }

                with torch.no_grad():
                    outputs = self.rerank_model(**batch_dict, return_dict=True)
                    logits = outputs.logits

                for j, logit in enumerate(logits):
                    if j < len(batch):
                        batch[j]["score"] = float(torch.sigmoid(logit).item())
                        reranked.append(batch[j])

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM в реранкере батч {i//batch_size + 1}, пропускаем...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            torch.cuda.empty_cache()

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:n_results]
