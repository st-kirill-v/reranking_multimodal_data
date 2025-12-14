"""
E5 семантический поиск (каскадный с BM25)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class E5Module:
    """E5 для каскадного поиска: BM25 → E5"""

    def __init__(
        self,
        name: str = "e5",
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
        bm25_module_name: str = "bm25",
        top_k_candidates: int = 100,
    ):
        # НЕ вызываем super().__init__() - BaseSearchModule требует переработки

        self.name = name
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bm25_module_name = bm25_module_name
        self.top_k_candidates = top_k_candidates

        # Модель
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # Кэш эмбеддингов
        self.doc_embeddings = {}  # doc_id -> embedding

        logger.info(f"Создан E5 модуль '{name}' (каскадный с {bm25_module_name})")

    def _load_model(self):
        if self._model_loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"Загружаю E5: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._model_loaded = True
        except ImportError:
            logger.error("Установите: pip install transformers torch")
            raise

    def _encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        if not self._model_loaded:
            self._load_model()

        # Префиксы для E5
        if is_query:
            text = f"query: {text}"
        else:
            text = f"passage: {text}"

        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        # Нормализация
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()[0]

    def add_documents(
        self, documents: List[str], ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Документы будут обработаны BM25, E5 эмбеддинги лениво"""
        logger.info(f"E5: готов к каскадному поиску с {len(documents)} документами")
        return {
            "status": "success",
            "message": "E5 готов, документы будут в BM25",
            "module": self.name,
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Каскадный поиск:
        1. Получить BM25 модуль из rag_engine
        2. BM25 находит кандидатов
        3. E5 переранжирует их
        """
        # 1. Получаем BM25 модуль
        bm25_module = self._get_bm25_module()
        if not bm25_module:
            logger.error(f"BM25 модуль '{self.bm25_module_name}' не найден")
            return []

        # 2. BM25: быстрый поиск кандидатов
        bm25_results = bm25_module.search(query, top_k=self.top_k_candidates)

        if not bm25_results:
            return []

        logger.info(f"BM25 нашел {len(bm25_results)} кандидатов, E5 переранжирует...")

        # 3. E5: кодируем запрос
        query_embedding = self._encode_text(query, is_query=True)

        # 4. Переранжирование
        reranked = []
        for candidate in bm25_results:
            doc_id = candidate.get("id")
            doc_text = candidate.get("content", "")

            # Получаем или вычисляем эмбеддинг
            if doc_id in self.doc_embeddings:
                doc_embedding = self.doc_embeddings[doc_id]
            else:
                doc_embedding = self._encode_text(doc_text, is_query=False)
                self.doc_embeddings[doc_id] = doc_embedding

            # Косинусное сходство
            similarity = float(np.dot(query_embedding, doc_embedding))

            # Комбинированный score
            bm25_score = candidate.get("score", 0)
            e5_score = (similarity + 1) / 2  # [-1,1] → [0,1]

            # Веса: 40% BM25 + 60% E5
            combined_score = 0.4 * bm25_score + 0.6 * e5_score

            reranked.append(
                {
                    "id": doc_id,
                    "content": doc_text,
                    "score": combined_score,
                    "bm25_score": bm25_score,
                    "e5_score": e5_score,
                    "e5_similarity": similarity,
                    "module": self.name,
                }
            )

        # 5. Сортировка
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # 6. Нормализация
        if reranked:
            scores = [r["score"] for r in reranked]
            max_score = max(scores) if max(scores) > 0 else 1.0

            for r in reranked:
                r["score"] = r["score"] / max_score

        return reranked[:top_k]

    def _get_bm25_module(self):
        """Получить BM25 модуль из системы"""
        try:
            from src.core.rag import rag_engine

            return rag_engine.manager.search_modules.get(self.bm25_module_name)
        except:
            logger.error("Не удалось получить BM25 модуль")
            return None

    def clear(self):
        self.doc_embeddings.clear()
        return {"status": "cleared", "name": self.name}

    def save(self, path: str):
        """Сохраняем эмбеддинги"""
        import os
        import pickle

        module_path = os.path.join(path, self.name)
        os.makedirs(module_path, exist_ok=True)

        save_data = {
            "doc_embeddings": self.doc_embeddings,
            "model_name": self.model_name,
            "bm25_module_name": self.bm25_module_name,
        }

        save_path = os.path.join(module_path, "e5_embeddings.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        return {"status": "saved"}

    def load(self, path: str) -> bool:
        import os
        import pickle

        save_path = os.path.join(path, self.name, "e5_embeddings.pkl")

        if not os.path.exists(save_path):
            return False

        try:
            with open(save_path, "rb") as f:
                save_data = pickle.load(f)

            self.doc_embeddings = save_data.get("doc_embeddings", {})
            return True
        except:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "e5",
            "name": self.name,
            "model": self.model_name,
            "bm25_source": self.bm25_module_name,
            "top_k_candidates": self.top_k_candidates,
            "embeddings_cached": len(self.doc_embeddings),
            "device": self.device,
        }
