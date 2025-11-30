import numpy as np
import faiss
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any


class SimpleTextRAG:
    def __init__(self, storage_path: str = "data/indexes"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # Инициализация компонентов
        self.vectorizer = TfidfVectorizer(max_features=1024, stop_words="english", lowercase=True)
        self.documents = []
        self.doc_ids = []
        self.is_fitted = False
        # FAISS индекс
        self.dimension = 1024
        self.index = faiss.IndexFlatIP(self.dimension)

    def add_documents(self, documents: List[str], ids: List[str] = None) -> Dict[str, Any]:
        """Добавить документы в систему"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Добавляем документы
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.doc_ids.extend(ids)

        # Переобучаем TF-IDF на всех документах
        if self.documents:
            tfidf_matrix = self.vectorizer.fit_transform(self.documents).toarray()

            # Пересоздаем индекс с новыми размерами
            self.index = faiss.IndexFlatIP(tfidf_matrix.shape[1])

            # Добавляем все эмбеддинги
            embeddings = tfidf_matrix.astype("float32")
            self.index.add(embeddings)
            self.is_fitted = True

            self._save_index()

        return {"status": "added", "count": len(documents)}

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Поиск по документам"""
        if not self.is_fitted or not self.documents:
            return {"query": query, "results": []}

        # Преобразуем запрос в вектор
        query_vector = self.vectorizer.transform([query]).toarray().astype("float32")

        # Ищем в FAISS
        distances, indices = self.index.search(query_vector, n_results)

        # Формируем результаты
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "document": self.documents[idx],
                        "score": float(distances[0][i]),
                    }
                )

        return {"query": query, "results": results}

    def get_info(self) -> Dict[str, Any]:
        """Информация о системе"""
        return {
            "total_documents": len(self.documents),
            "is_fitted": self.is_fitted,
            "embedding_dimension": self.dimension,
        }

    def clear_documents(self) -> Dict[str, Any]:
        """Очистить все документы"""
        self.vectorizer = TfidfVectorizer(max_features=1024, stop_words="english", lowercase=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.doc_ids = []
        self.is_fitted = False

        # Удаляем файлы индекса
        index_files = ["index.faiss", "documents.json", "mapping.json"]
        for file in index_files:
            path = os.path.join(self.storage_path, file)
            if os.path.exists(path):
                os.remove(path)

        return {"status": "cleared"}

    def _save_index(self):
        """Сохранить индекс на диск"""
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, os.path.join(self.storage_path, "index.faiss"))

        # Сохраняем документы и маппинг
        data = {"documents": self.documents, "doc_ids": self.doc_ids, "is_fitted": self.is_fitted}

        with open(os.path.join(self.storage_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_index(self):
        """Загрузить индекс с диска"""
        index_path = os.path.join(self.storage_path, "index.faiss")
        documents_path = os.path.join(self.storage_path, "documents.json")

        if os.path.exists(index_path) and os.path.exists(documents_path):
            # Загружаем FAISS индекс
            self.index = faiss.read_index(index_path)

            # Загружаем документы
            with open(documents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.doc_ids = data["doc_ids"]
                self.is_fitted = data["is_fitted"]

            return True
        return False


# Глобальный экземпляр RAG
rag_engine = SimpleTextRAG()
