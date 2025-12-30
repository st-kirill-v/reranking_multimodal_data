"""
bm25 модуль для поиска.
реализация bm25 с базовой предобработкой текста.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json
import pickle
import os
import nltk
from src.core.base import BaseSearchModule


class BM25Module(BaseSearchModule):
    def __init__(
        self, name: str = "bm25", language: str = "multilingual", k1: float = 2.5, b: float = 0.9
    ):
        self.name = name
        self.language = language
        self.k1 = k1
        self.b = b
        self.is_fitted = False
        self.documents = []
        self.total_terms = 0
        self.ids = []
        self.bm25 = None

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        self.stop_words = set()
        for lang in ["english", "russian", "french", "spanish", "german"]:
            try:
                self.stop_words.update(stopwords.words(lang))
            except:
                pass

        important_words = {
            "war",
            "world",
            "technology",
            "python",
            "intelligence",
            "bitcoin",
            "blockchain",
            "искусственный",
            "интеллект",
            "технология",
        }
        self.stop_words = {w for w in self.stop_words if w not in important_words}

    def _preprocess_text(self, text: str) -> List[str]:
        try:
            tokens = nltk.word_tokenize(text.lower())
        except:
            import re

            tokens = re.findall(r"\b\w+\b", text.lower())

        processed = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words and not token.isdigit():
                processed.append(token)

        return processed

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        if ids is None:
            ids = [
                f"{self.name}_{i}"
                for i in range(len(self.documents), len(self.documents) + len(documents))
            ]

        if self.documents:
            self.documents = []
            self.ids = []

        self.documents.extend(documents)
        self.ids.extend(ids)

        processed_docs = [self._preprocess_text(doc) for doc in self.documents]

        if not processed_docs or all(len(doc) == 0 for doc in processed_docs):
            print(f"{self.name}: все документы пустые после предобработки")
            self.is_fitted = False
            return {
                "module": self.name,
                "status": "error",
                "message": "all documents empty after preprocessing",
            }

        from rank_bm25 import BM25Okapi

        try:
            # ИСПОЛЬЗУЕМ self.k1 и self.b вместо жестко заданных значений
            self.bm25 = BM25Okapi(processed_docs, k1=self.k1, b=self.b)
            self.is_fitted = True
            self.total_terms = sum(len(doc) for doc in processed_docs)

            print(f"{self.name}: индекс построен с k1={self.k1}, b={self.b}")  # ← ОБНОВЛЕНО

            return {
                "module": self.name,
                "status": "success",
                "added": len(documents),
                "total": len(self.documents),
                "total_terms": self.total_terms,
            }

        except Exception as e:
            print(f"{self.name}: ошибка построения индекса: {e}")
            self.is_fitted = False
            return {"module": self.name, "status": "error", "message": str(e)}

    def fit(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        return self.add_documents(documents, ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_fitted or not self.bm25 or len(self.documents) == 0:
            print(f"{self.name}: поиск невозможен")
            return []

        try:
            processed_query = self._preprocess_text(query)

            if not processed_query:
                print(f"{self.name}: запрос '{query}' пустой после предобработки")
                return []

            raw_scores = self.bm25.get_scores(processed_query)

            if len(raw_scores) > 0:
                min_score = np.min(raw_scores)
                max_score = np.max(raw_scores)

                if max_score - min_score < 1e-6:
                    normalized_scores = np.ones_like(raw_scores) * 0.5
                else:
                    normalized_scores = (raw_scores - min_score) / (max_score - min_score)

                scores = normalized_scores
            else:
                scores = np.array([])

            if len(scores) > 0 and np.max(scores) > 0:
                top_indices = np.argsort(scores)[::-1][:top_k]

                results = []
                for idx in top_indices:
                    if scores[idx] > 0.01:
                        results.append(
                            {
                                "id": self.ids[idx],
                                "content": self.documents[idx],
                                "score": float(scores[idx]),
                                "raw_score": float(raw_scores[idx]),
                                "module": self.name,
                                "module_type": "bm25",
                            }
                        )

                print(f"{self.name}: поиск '{query}' -> {len(results)} результатов")
                return results
            else:
                top_indices = np.argsort(raw_scores)[::-1][: min(top_k, 3)]
                results = []
                for idx in top_indices:
                    results.append(
                        {
                            "id": self.ids[idx],
                            "content": self.documents[idx],
                            "score": 0.05,
                            "raw_score": float(raw_scores[idx]),
                            "module": self.name,
                            "module_type": "bm25",
                            "note": "low_confidence",
                        }
                    )

                if results:
                    print(f"{self.name}: низкие скоры для '{query}'")
                    return results
                else:
                    return []

        except Exception as e:
            print(f"{self.name}: ошибка поиска: {e}")
            return []

    def get_info(self):
        return {
            "name": self.name,
            "type": "bm25",
            "language": self.language,
            "k1": self.k1,
            "b": self.b,
            "total_documents": len(self.documents),
            "total_terms": self.total_terms,
            "is_fitted": self.is_fitted,
        }

    def clear(self) -> Dict:
        self.documents = []
        self.ids = []
        self.bm25 = None
        self.is_fitted = False
        self.total_terms = 0
        return {"module": self.name, "status": "cleared"}

    def save(self, path: str):
        module_path = os.path.join(path, self.name)
        os.makedirs(module_path, exist_ok=True)

        data = {
            "documents": self.documents,
            "ids": self.ids,
            "language": self.language,
            "k1": self.k1,  # ← ДОБАВЛЕНО для сохранения
            "b": self.b,  # ← ДОБАВЛЕНО для сохранения
            "total_terms": self.total_terms,
            "is_fitted": self.is_fitted,
        }

        with open(os.path.join(module_path, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if self.bm25:
            with open(os.path.join(module_path, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)

    def load(self, path: str) -> Dict[str, Any]:  # ← Изменили возвращаемый тип
        module_path = os.path.join(path, self.name)

        if not os.path.exists(module_path):
            return {"status": "not_found"}  # ← Возвращаем dict, а не False

        try:
            with open(os.path.join(module_path, "data.json"), "r", encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data["documents"]
            self.ids = data["ids"]
            self.language = data.get("language", "multilingual")
            self.k1 = data.get("k1", 2.5)
            self.b = data.get("b", 0.9)
            self.total_terms = data.get("total_terms", 0)
            self.is_fitted = data.get("is_fitted", False)

            bm25_path = os.path.join(module_path, "bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)

            print(
                f"{self.name}: загружено {len(self.documents)} документов (k1={self.k1}, b={self.b})"
            )
            return {"status": "loaded", "count": len(self.documents)}  # ← Всегда возвращаем dict

        except Exception as e:
            print(f"ошибка загрузки модуля {self.name}: {e}")
            return {"status": "error", "error": str(e)}
