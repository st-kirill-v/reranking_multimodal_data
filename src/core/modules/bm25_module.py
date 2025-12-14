"""
bm25_module.py - BM25 модуль для поиска (мультиязычный)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json
import pickle
import os
import nltk
from src.core.base import BaseSearchModule


class BM25Module(BaseSearchModule):

    def __init__(self, name: str = "bm25", language: str = "multilingual"):
        self.name = name
        self.language = language
        self.is_fitted = False
        self.documents = []
        self.total_terms = 0
        self.ids = []
        self.bm25 = None

        # Загружаем стоп-слова для нескольких языков
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        # Объединяем стоп-слова из разных языков
        self.stop_words = set()
        for lang in ["english", "russian", "french", "spanish", "german"]:
            try:
                self.stop_words.update(stopwords.words(lang))
            except:
                pass

        # Убираем важные слова из стоп-слов
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
        """Предобработка без стемминга"""
        try:
            tokens = nltk.word_tokenize(text.lower())
        except:
            import re

            tokens = re.findall(r"\b\w+\b", text.lower())

        # Более мягкая фильтрация
        processed = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words and not token.isdigit():
                processed.append(token)

        return processed

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        """Добавляет документы и строит индекс"""
        if ids is None:
            ids = [
                f"{self.name}_{i}"
                for i in range(len(self.documents), len(self.documents) + len(documents))
            ]

        # Если документы уже есть, очищаем
        if self.documents:
            self.documents = []
            self.ids = []

        # Сохраняем оригинальные документы
        self.documents.extend(documents)
        self.ids.extend(ids)

        # Обрабатываем все документы
        processed_docs = [self._preprocess_text(doc) for doc in self.documents]

        # Проверяем что есть обработанные документы
        if not processed_docs or all(len(doc) == 0 for doc in processed_docs):
            print(f"⚠️ {self.name}: Все документы пустые после предобработки")
            self.is_fitted = False
            return {
                "module": self.name,
                "status": "error",
                "message": "All documents empty after preprocessing",
            }

        # Создаем/обновляем BM25 индекс
        from rank_bm25 import BM25Okapi

        try:
            # 🔥 ШАГ 2: ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ БОЛЬШОЙ КОЛЛЕКЦИИ
            self.bm25 = BM25Okapi(
                processed_docs,
                k1=2.5,  # Увеличить для больших коллекций (было 1.2)
                b=0.9,  # Увеличить для лучшего учета длины документа
            )
            self.is_fitted = True
            self.total_terms = sum(len(doc) for doc in processed_docs)

            print(f"✅ {self.name}: Индекс построен с k1=2.5, b=0.9")

            return {
                "module": self.name,
                "status": "success",
                "added": len(documents),
                "total": len(self.documents),
                "total_terms": self.total_terms,
            }

        except Exception as e:
            print(f"❌ {self.name}: Ошибка построения индекса: {e}")
            self.is_fitted = False
            return {"module": self.name, "status": "error", "message": str(e)}

    def fit(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        """Алиас для add_documents (совместимость с другими модулями)"""
        return self.add_documents(documents, ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск по запросу"""
        if not self.is_fitted or not self.bm25 or len(self.documents) == 0:
            print(
                f"⚠️ {self.name}: Поиск невозможен. is_fitted={self.is_fitted}, bm25={self.bm25 is not None}, docs={len(self.documents)}"
            )
            return []

        try:
            # Обрабатываем запрос
            processed_query = self._preprocess_text(query)

            # Проверяем что запрос не пустой после предобработки
            if not processed_query:
                print(f"⚠️ {self.name}: Запрос '{query}' пустой после предобработки")
                return []

            # Получаем скоры
            raw_scores = self.bm25.get_scores(processed_query)

            # 🔥 ШАГ 1: ПРОСТАЯ НОРМАЛИЗАЦИЯ
            if len(raw_scores) > 0:
                # Получаем min и max
                min_score = np.min(raw_scores)
                max_score = np.max(raw_scores)

                # Если все scores одинаковые (редкий случай)
                if max_score - min_score < 1e-6:
                    normalized_scores = np.ones_like(raw_scores) * 0.5
                else:
                    # Простая min-max нормализация к [0, 1]
                    normalized_scores = (raw_scores - min_score) / (max_score - min_score)

                scores = normalized_scores
            else:
                scores = np.array([])

            # Проверяем что есть хотя бы один положительный скор
            if len(scores) > 0 and np.max(scores) > 0:
                # Сортируем по убыванию
                top_indices = np.argsort(scores)[::-1][:top_k]

                # Формируем результаты
                results = []
                for idx in top_indices:
                    if scores[idx] > 0.01:  # Небольшой порог
                        results.append(
                            {
                                "id": self.ids[idx],
                                "content": self.documents[idx],
                                "score": float(scores[idx]),  # Нормализованный скор 0-1
                                "raw_score": float(raw_scores[idx]),
                                "module": self.name,
                                "module_type": "bm25",
                            }
                        )

                print(f"✅ {self.name}: Поиск '{query}' -> {len(results)} результатов")
                return results
            else:
                # Fallback: показываем топ-3 с минимальными скорами
                top_indices = np.argsort(raw_scores)[::-1][: min(top_k, 3)]
                results = []
                for idx in top_indices:
                    results.append(
                        {
                            "id": self.ids[idx],
                            "content": self.documents[idx],
                            "score": 0.05,  # Минимальный confidence
                            "raw_score": float(raw_scores[idx]),
                            "module": self.name,
                            "module_type": "bm25",
                            "note": "low_confidence",
                        }
                    )

                if results:
                    print(f"⚠️ {self.name}: Низкие скоры для '{query}'")
                    return results
                else:
                    return []

        except Exception as e:
            print(f"❌ {self.name}: Ошибка поиска: {e}")
            return []

    def get_info(self):
        return {
            "name": self.name,
            "type": "bm25",
            "language": self.language,
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
        """Сохраняем состояние модуля"""
        module_path = os.path.join(path, self.name)
        os.makedirs(module_path, exist_ok=True)

        # Сохраняем данные
        data = {
            "documents": self.documents,
            "ids": self.ids,
            "language": self.language,
            "total_terms": self.total_terms,
            "is_fitted": self.is_fitted,
        }

        with open(os.path.join(module_path, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Сохраняем BM25 модель (если есть)
        if self.bm25:
            with open(os.path.join(module_path, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)

    def load(self, path: str) -> bool:
        """Загружаем состояние модуля"""
        module_path = os.path.join(path, self.name)

        if not os.path.exists(module_path):
            return False

        try:
            # Загружаем данные
            with open(os.path.join(module_path, "data.json"), "r", encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data["documents"]
            self.ids = data["ids"]
            self.language = data.get("language", "multilingual")
            self.total_terms = data.get("total_terms", 0)
            self.is_fitted = data.get("is_fitted", False)

            # Загружаем BM25 модель
            bm25_path = os.path.join(module_path, "bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)

            print(
                f"✅ {self.name}: Загружено {len(self.documents)} документов, is_fitted={self.is_fitted}"
            )
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модуля {self.name}: {e}")
            return False
