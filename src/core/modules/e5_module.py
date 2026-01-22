"""
модуль семантического поиска на основе e5 эмбеддингов с поддержкой каскадного поиска bm25-e5.
"""

import numpy as np
from typing import List, Dict, Any, Optional
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
        manager=None,
        storage_path: str = "cache/modules",
    ):
        self.name = name
        self.model_path = model_path
        self.bm25_module_name = bm25_module_name
        self.top_k_candidates = top_k_candidates
        self.manager = manager
        self.storage_path = storage_path
        self.doc_embeddings = {}
        self.documents = []
        self.is_fitted = False

        # Проверяем путь к модели
        if not os.path.exists(model_path):
            print(f"Предупреждение: модель не найдена по пути {model_path}")
            print("Попытка загрузки из Hugging Face Hub...")
            self.model_path = "intfloat/multilingual-e5-small"

        self._load_model(device)
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
            return np.zeros(384)

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

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()[0]

        except Exception as e:
            print(f"Ошибка кодирования текста: {e}")
            return np.zeros(384)

    def fit(self, documents: List[str]):
        """Совместимость с интерфейсом BM25Module."""
        print(f"E5Module: fit для {len(documents)} документов")
        self.documents = documents.copy()

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

    def add_documents(
        self, documents: List[str], ids: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:

        print(f"E5Module.add_documents ВЫЗВАН!")
        print(f"Модуль: {self.name}")
        print(f"Получаю документов: {len(documents)}")
        print(f"IDs: {ids}")

        start_idx = len(self.documents)
        self.documents.extend(documents)

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
        self.save()

        return {
            "status": "success",
            "added": len(documents),
            "total_documents": len(self.documents),
            "embeddings_cached": cached,
            "name": self.name,
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Поиск с использованием BM25 + E5 реранжирования."""
        if not query or not query.strip():
            return []

        print(f"E5Module: поиск '{query}'")

        # 1. Получаем BM25 модуль через manager
        bm25_module = None
        if self.manager and self.bm25_module_name:
            bm25_module = self.manager.search_modules.get(self.bm25_module_name)

        if not bm25_module:
            print(f"E5Module: BM25 модуль '{self.bm25_module_name}' не найден")
            # Fallback: семантический поиск без BM25
            return self.semantic_search(query, top_k)

        # 2. Получаем результаты BM25
        try:
            bm25_results = bm25_module.search(query, top_k=self.top_k_candidates)
        except Exception as e:
            print(f"E5Module: ошибка BM25 поиска: {e}")
            return self.semantic_search(query, top_k)

        if not bm25_results:
            print(f"E5Module: BM25 не вернул результатов")
            return self.semantic_search(query, top_k)

        print(f"E5Module: BM25 вернул {len(bm25_results)} кандидатов")

        # 3. Кодируем запрос
        try:
            query_embedding = self._encode_text(query, is_query=True)
        except Exception as e:
            print(f"E5Module: ошибка кодирования запроса: {e}")
            return bm25_results[:top_k]

        # 4. Реранжируем с E5
        reranked = []
        for i, candidate in enumerate(bm25_results):
            doc_id = candidate.get("id", f"doc_{i}")
            doc_text = candidate.get("content", "")

            if not doc_text.strip():
                continue

            if doc_id in self.doc_embeddings:
                doc_embedding = self.doc_embeddings[doc_id]
            else:
                try:
                    doc_embedding = self._encode_text(doc_text, is_query=False)
                    self.doc_embeddings[doc_id] = doc_embedding
                except Exception as e:
                    print(f"E5Module: ошибка кодирования документа {doc_id}: {e}")
                    continue

            try:
                similarity = float(np.dot(query_embedding, doc_embedding))
                e5_score = (similarity + 1) / 2
            except Exception as e:
                print(f"E5Module: ошибка вычисления схожести: {e}")
                e5_score = 0.5

            bm25_score = candidate.get("score", 0)
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

        reranked.sort(key=lambda x: x["score"], reverse=True)

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

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback: семантический поиск без BM25."""
        if not query or not query.strip():
            return []

        print(f"E5Module: семантический поиск '{query}'")

        try:
            query_embedding = self._encode_text(query, is_query=True)
        except Exception as e:
            print(f"E5Module: ошибка кодирования запроса: {e}")
            return []

        results = []
        for doc_id, doc_embedding in self.doc_embeddings.items():
            try:
                similarity = float(np.dot(query_embedding, doc_embedding))
                score = (similarity + 1) / 2

                if doc_id.startswith("doc_"):
                    try:
                        idx = int(doc_id[4:])
                        content = (
                            self.documents[idx]
                            if idx < len(self.documents)
                            else f"Документ {doc_id}"
                        )
                    except:
                        content = f"Документ {doc_id}"
                else:
                    content = f"Документ {doc_id}"

                results.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "score": score,
                        "similarity": similarity,
                        "module": self.name,
                        "method": "e5_semantic",
                    }
                )
            except Exception as e:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Сохраняет кешированные эмбеддинги. Не падает при ошибках."""
        # Если нет эмбеддингов - нечего сохранять
        if not self.doc_embeddings:
            return {"status": "skipped", "reason": "no_embeddings"}

        # Определяем путь
        if path is None:
            if hasattr(self, "storage_path") and self.storage_path:
                target_dir = self.storage_path
                # Убедимся что папка существует (пробуем создать)
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except PermissionError:
                    # Если нет прав, используем текущую директорию
                    target_dir = "."
                    print(f"E5Module: нет прав на {self.storage_path}, сохраняю в {target_dir}")
            else:
                target_dir = "."

            path = os.path.join(target_dir, f"{self.name}_embeddings.npz")

        print(f"E5Module: пытаюсь сохранить {len(self.doc_embeddings)} эмбеддингов в {path}")

        # Пытаемся сохранить
        try:
            if self.doc_embeddings:
                embeddings_dict = {}
                metadata = {}

                for doc_id, embedding in self.doc_embeddings.items():
                    embeddings_dict[doc_id] = embedding
                    if doc_id.startswith("doc_"):
                        try:
                            idx = int(doc_id[4:])
                            if idx < len(self.documents):
                                metadata[doc_id] = self.documents[idx][:200]
                        except:
                            pass

                np.savez_compressed(
                    path,
                    **embeddings_dict,
                    __metadata=json.dumps(metadata),
                    __doc_count=len(self.documents),
                )

            print(f"E5Module: успешно сохранено в {path}")
            return {"status": "saved", "path": path, "embeddings_count": len(self.doc_embeddings)}

        except PermissionError as e:
            print(f"E5Module: нет прав на сохранение в {path}")
            return {"status": "permission_error", "message": str(e)}

        except Exception as e:
            print(f"E5Module: ошибка сохранения: {e}")
            return {"status": "error", "message": str(e)}

    def _load_embeddings(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Загружает кешированные эмбеддинги. Не падает при ошибках."""
        # 1. Определяем путь
        if path is None:
            # Пробуем storage_path, если он есть
            if hasattr(self, "storage_path") and self.storage_path:
                target_dir = self.storage_path
            else:
                # Fallback: рядом с моделью
                target_dir = os.path.dirname(self.model_path)
                if not os.path.exists(target_dir):
                    target_dir = "."  # Текущая директория

            path = os.path.join(target_dir, f"{self.name}_embeddings.npz")

        print(f"E5Module: пытаюсь загрузить эмбеддинги из {path}")

        # 2. Проверяем существует ли файл
        if not os.path.exists(path):
            print(f"E5Module: файл эмбеддингов не найден: {path}")
            self.doc_embeddings = {}
            return {"status": "not_found", "path": path}

        # 3. Пытаемся загрузить (с защитой от ошибок)
        try:
            data = np.load(path, allow_pickle=True)
            loaded_count = 0
            self.doc_embeddings = {}  # Очищаем перед загрузкой

            if "__doc_count" in data:
                try:
                    doc_count = int(data["__doc_count"])
                    print(f"E5Module: в файле было {doc_count} документов")
                    # Создаем список документов нужной длины
                    self.documents = [""] * doc_count
                except Exception as e:
                    print(f"E5Module: ошибка загрузки doc_count: {e}")

            if "__metadata" in data:
                try:
                    metadata = json.loads(str(data["__metadata"]))
                    metadata_loaded = 0
                    for doc_id, text_snippet in metadata.items():
                        # doc_id выглядит как "doc_123"
                        try:
                            idx = int(doc_id[4:])  # убираем "doc_"
                            if idx < len(self.documents):
                                self.documents[idx] = text_snippet
                                metadata_loaded += 1
                        except:
                            continue
                    if metadata_loaded > 0:
                        print(f"E5Module: загружено {metadata_loaded} текстов документов")
                except Exception as e:
                    print(f"E5Module: ошибка загрузки метаданных: {e}")

            if self.documents:
                self.is_fitted = True

            # Загружаем эмбеддинги
            for key in data.files:
                if not key.startswith("__"):
                    self.doc_embeddings[key] = data[key]
                    loaded_count += 1

            print(f"E5Module: успешно загружено {loaded_count} эмбеддингов")
            return {"status": "loaded", "count": loaded_count, "documents": len(self.documents)}

        except PermissionError as e:
            # Специальная обработка ошибки прав доступа
            print(f"E5Module: нет прав доступа к файлу {path}")
            print(f"  Ошибка: {e}")
            self.doc_embeddings = {}
            return {"status": "permission_error", "message": str(e)}

        except OSError as e:
            # Ошибки файловой системы
            print(f"E5Module: ошибка доступа к файлу: {e}")
            self.doc_embeddings = {}
            return {"status": "io_error", "message": str(e)}

        except Exception as e:
            # Любая другая ошибка
            print(f"E5Module: ошибка загрузки эмбеддингов: {e}")
            print(f"  Тип ошибки: {type(e).__name__}")
            self.doc_embeddings = {}
            return {"status": "error", "message": str(e), "type": type(e).__name__}

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Загружает эмбеддинги."""
        # Если path - это папка (не файл), корректируем
        if path and os.path.isdir(path):
            # Это папка, делаем путь к файлу
            corrected_path = os.path.join(path, f"{self.name}_embeddings.npz")
            print(f"E5Module: исправляю путь {path} → {corrected_path}")
            return self._load_embeddings(corrected_path)

        # Иначе передаем как есть
        return self._load_embeddings(path)

    def clear(self) -> Dict[str, Any]:
        self.doc_embeddings.clear()
        self.documents = []
        self.is_fitted = False

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
        return len(self.documents)

    def build_index(self, **kwargs):
        """Строит индекс эмбеддингов."""
        print(f"E5Module.build_index: начинаю, документов: {len(self.documents)}")

        if not self.documents:
            print("E5Module.build_index: нет документов!")
            return {"status": "error", "message": "No documents"}

        computed = 0
        for i, doc in enumerate(self.documents):
            doc_id = f"doc_{i}"
            if doc_id not in self.doc_embeddings and doc.strip():
                try:
                    embedding = self._encode_text(doc, is_query=False)
                    self.doc_embeddings[doc_id] = embedding
                    computed += 1
                except Exception as e:
                    print(f"E5Module.build_index: ошибка для документа {i}: {e}")

        save_result = self.save()
        self.is_fitted = True

        print(f"E5Module.build_index: готово! Эмбеддингов: {len(self.doc_embeddings)}")

        return {
            "status": "success",
            "documents": len(self.documents),
            "embeddings": len(self.doc_embeddings),
            "computed": computed,
            "save_result": save_result,
        }

    def get_info(self) -> Dict[str, Any]:
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
