"""
модуль семантического поиска на основе e5 эмбеддингов с поддержкой каскадного поиска bm25-e5.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import json


class E5Module:
    def __init__(
        self,
        name: str = "e5",
        model_path: str = "./models/e5/e5-small-v2",
        bm25_module_name: str = "bm25",
        top_k_candidates: int = 100,
        device: Optional[str] = None,
    ):
        self.name = name
        self.model_path = model_path
        self.bm25_module_name = bm25_module_name
        self.top_k_candidates = top_k_candidates
        self.doc_embeddings = {}
        self.documents = []  # Добавляем для совместимости
        self.is_fitted = False  # Добавляем для совместимости

        # Проверяем путь к модели
        if not os.path.exists(model_path):
            print(f"Предупреждение: модель не найдена по пути {model_path}")
            print("Попытка загрузки из Hugging Face Hub...")
            self.model_path = "intfloat/multilingual-e5-small"

        self._load_model(device)

        # Загружаем сохраненные эмбеддинги если есть
        self._load_embeddings()

    def _load_model(self, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        print(f"E5Module: загрузка модели из {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            # Пробуем загрузить локально
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            except Exception as e2:
                raise RuntimeError(f"Не удалось загрузить модель: {e2}")

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"E5Module: модель загружена на устройство {self.device}")

    def _encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        import torch

        if not text or not text.strip():
            return np.zeros(384)  # Размерность e5-small

        if is_query:
            text = f"query: {text}"
        else:
            text = f"passage: {text}"

        try:
            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Средний пулинг с учетом маски внимания
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

            # Нормализация
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()[0]

        except Exception as e:
            print(f"Ошибка кодирования текста: {e}")
            return np.zeros(384)  # Fallback

    def fit(self, documents: List[str]):
        """Совместимость с интерфейсом BM25Module."""
        print(f"E5Module: fit для {len(documents)} документов")

        self.documents = documents.copy()

        # Предварительно кешируем эмбеддинги для всех документов
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            if doc_id not in self.doc_embeddings and doc.strip():
                try:
                    embedding = self._encode_text(doc, is_query=False)
                    self.doc_embeddings[doc_id] = embedding
                except Exception as e:
                    print(f"Ошибка кодирования документа {i}: {e}")

        self.is_fitted = True
        return {"status": "fitted", "count": len(documents)}

    def add_documents(self, documents: List[str], **kwargs) -> Dict[str, Any]:
        """Добавляет документы и кеширует их эмбеддинги."""
        print(f"E5Module: добавление {len(documents)} документов")

        start_idx = len(self.documents)
        self.documents.extend(documents)

        # Кешируем эмбеддинги новых документов
        cached = 0
        for i, doc in enumerate(documents):
            if not doc.strip():
                continue

            doc_id = f"doc_{start_idx + i}"
            try:
                embedding = self._encode_text(doc, is_query=False)
                self.doc_embeddings[doc_id] = embedding
                cached += 1
            except Exception as e:
                print(f"Ошибка кодирования документа {start_idx + i}: {e}")

        self.is_fitted = True

        # Сохраняем эмбеддинги
        self.save()

        return {
            "status": "added",
            "total_documents": len(self.documents),
            "embeddings_cached": cached,
            "name": self.name,
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Поиск с использованием BM25 + E5 реранжирования."""
        from src.core.rag import rag_engine

        if not query or not query.strip():
            return []

        print(f"E5Module: поиск '{query}'")

        # 1. Получаем BM25 результаты
        bm25_module = rag_engine.manager.search_modules.get(self.bm25_module_name)
        if not bm25_module:
            print(f"E5Module: BM25 модуль '{self.bm25_module_name}' не найден")
            return []

        bm25_results = bm25_module.search(query, top_k=self.top_k_candidates)

        if not bm25_results:
            print(f"E5Module: BM25 не вернул результатов")
            return []

        print(f"E5Module: BM25 вернул {len(bm25_results)} кандидатов")

        # 2. Кодируем запрос
        try:
            query_embedding = self._encode_text(query, is_query=True)
        except Exception as e:
            print(f"E5Module: ошибка кодирования запроса: {e}")
            return bm25_results[:top_k]  # Fallback к BM25

        # 3. Реранжируем с E5
        reranked = []
        for i, candidate in enumerate(bm25_results):
            doc_id = candidate.get("id", f"doc_{i}")
            doc_text = candidate.get("content", "")

            if not doc_text.strip():
                continue

            # Получаем или вычисляем эмбеддинг документа
            if doc_id in self.doc_embeddings:
                doc_embedding = self.doc_embeddings[doc_id]
            else:
                try:
                    doc_embedding = self._encode_text(doc_text, is_query=False)
                    self.doc_embeddings[doc_id] = doc_embedding
                except Exception as e:
                    print(f"E5Module: ошибка кодирования документа {doc_id}: {e}")
                    continue

            # Вычисляем схожесть
            try:
                similarity = float(np.dot(query_embedding, doc_embedding))
                e5_score = (similarity + 1) / 2  # Нормализация к [0, 1]
            except Exception as e:
                print(f"E5Module: ошибка вычисления схожести: {e}")
                e5_score = 0.5

            bm25_score = candidate.get("score", 0)

            # Комбинированная оценка
            combined_score = 0.3 * bm25_score + 0.7 * e5_score

            reranked.append(
                {
                    "id": doc_id,
                    "content": doc_text,
                    "score": combined_score,
                    "bm25_score": bm25_score,
                    "e5_score": e5_score,
                    "similarity": similarity,
                    "module": self.name,
                    "method": "bm25+e5",
                }
            )

        if not reranked:
            return []

        # Сортируем по комбинированной оценке
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # Нормализация оценок
        if reranked:
            scores = [r["score"] for r in reranked]
            max_score = max(scores)
            min_score = min(scores)

            if max_score > min_score:
                for r in reranked:
                    r["score"] = (r["score"] - min_score) / (max_score - min_score)
            else:
                for r in reranked:
                    r["score"] = 1.0

        print(f"E5Module: реранжировано {len(reranked)} документов")
        return reranked[:top_k]

    def save(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Сохраняет кешированные эмбеддинги."""
        if path is None:
            # Используем путь рядом с моделью
            save_dir = os.path.dirname(self.model_path)
            if not os.path.exists(save_dir):
                save_dir = "data/e5_cache"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{self.name}_embeddings.npz")

        try:
            # Сохраняем эмбеддинги
            if self.doc_embeddings:
                embeddings_dict = {}
                metadata = {}

                for doc_id, embedding in self.doc_embeddings.items():
                    embeddings_dict[doc_id] = embedding
                    # Извлекаем индекс из doc_id
                    if doc_id.startswith("doc_"):
                        try:
                            idx = int(doc_id[4:])
                            if idx < len(self.documents):
                                metadata[doc_id] = self.documents[idx][:200]  # Сохраняем превью
                        except:
                            pass

                # Сохраняем numpy массивы
                np.savez_compressed(
                    path,
                    **embeddings_dict,
                    __metadata=json.dumps(metadata),
                    __doc_count=len(self.documents),
                )

            return {
                "status": "saved",
                "path": path,
                "embeddings_count": len(self.doc_embeddings),
                "documents_count": len(self.documents),
            }

        except Exception as e:
            print(f"E5Module: ошибка сохранения: {e}")
            return {"status": "error", "message": str(e)}

    def _load_embeddings(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Загружает кешированные эмбеддинги."""
        if path is None:
            # Ищем файл рядом с моделью
            save_dir = os.path.dirname(self.model_path)
            if not os.path.exists(save_dir):
                save_dir = "data/e5_cache"
            path = os.path.join(save_dir, f"{self.name}_embeddings.npz")

        if not os.path.exists(path):
            print(f"E5Module: файл эмбеддингов не найден: {path}")
            return {"status": "not_found"}

        try:
            data = np.load(path, allow_pickle=True)
            loaded_count = 0

            for key in data.files:
                if not key.startswith("__"):
                    self.doc_embeddings[key] = data[key]
                    loaded_count += 1

            # Восстанавливаем метаданные
            if "__metadata" in data:
                metadata = json.loads(str(data["__metadata"]))
                print(f"E5Module: загружено {loaded_count} эмбеддингов")

            if "__doc_count" in data:
                doc_count = int(data["__doc_count"])
                print(f"E5Module: в кеше {doc_count} документов")

            return {"status": "loaded", "count": loaded_count}

        except Exception as e:
            print(f"E5Module: ошибка загрузки эмбеддингов: {e}")
            return {"status": "error", "message": str(e)}

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Совместимость с интерфейсом Module."""
        return self._load_embeddings(path)

    def clear(self) -> Dict[str, Any]:
        """Очищает все кешированные данные."""
        self.doc_embeddings.clear()
        self.documents = []
        self.is_fitted = False

        # Удаляем файл кеша если существует
        save_dir = os.path.dirname(self.model_path)
        if not os.path.exists(save_dir):
            save_dir = "data/e5_cache"
        cache_path = os.path.join(save_dir, f"{self.name}_embeddings.npz")

        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except:
                pass

        return {"status": "cleared", "name": self.name, "embeddings_cleared": True}

    def get_document_count(self) -> int:
        """Возвращает количество документов."""
        return len(self.documents)

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модуле."""
        return {
            "type": "e5",
            "name": self.name,
            "model": self.model_path,
            "device": self.device,
            "documents": len(self.documents),
            "embeddings_cached": len(self.doc_embeddings),
            "is_fitted": self.is_fitted,
            "bm25_source": self.bm25_module_name,
            "top_k_candidates": self.top_k_candidates,
        }
