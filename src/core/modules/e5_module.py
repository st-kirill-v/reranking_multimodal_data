"""
E5 модуль для TensorFlow (работает без PyTorch DLL проблем)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Автоматически определяем бэкенд
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch доступен, используем его")
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("⚠️  PyTorch недоступен, переключаемся на TensorFlow")

if not TORCH_AVAILABLE:
    import tensorflow as tf
    from transformers import TFAutoModel, AutoTokenizer

    logger.info("✅ TensorFlow импортирован")


class E5Module:
    """E5 семантический поиск с поддержкой PyTorch/TensorFlow"""

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
        self.device = device

        # Проверяем модель
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        # Загружаем модель
        self._load_model()

        # Кэш эмбеддингов
        self.doc_embeddings = {}

        logger.info(
            f"Создан E5 модуль '{name}' (бэкенд: {'PyTorch' if TORCH_AVAILABLE else 'TensorFlow'})"
        )

    def _load_model(self):
        """Загрузка модели с автоматическим выбором бэкенда"""
        logger.info(f"Загружаю модель E5 из {self.model_path}")

        try:
            if TORCH_AVAILABLE:
                # PyTorch версия
                import torch

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
                if self.device:
                    self.model = self.model.to(self.device)
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model = self.model.to(self.device)

                self.model.eval()
                logger.info(f"✅ PyTorch модель загружена на {self.device}")

            else:
                # TensorFlow версия
                import tensorflow as tf

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True
                )
                self.model = TFAutoModel.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    from_pt=True,  # Конвертируем веса PyTorch → TensorFlow
                )

                # Для TensorFlow автоматически используем GPU если доступно
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    self.device = "gpu"
                    logger.info(f"✅ TensorFlow модель загружена, GPU доступен")
                else:
                    self.device = "cpu"
                    logger.info(f"✅ TensorFlow модель загружена на CPU")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def _encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """Кодирование текста в эмбеддинг"""
        # E5 требует префиксы
        if is_query:
            text = f"query: {text}"
        else:
            text = f"passage: {text}"

        if TORCH_AVAILABLE:
            # PyTorch версия
            import torch

            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
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

        else:
            # TensorFlow версия
            import tensorflow as tf

            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="tf",  # Важно: return_tensors="tf"
            )

            outputs = self.model(inputs)

            # Mean pooling для TensorFlow
            token_embeddings = outputs.last_hidden_state
            attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
            input_mask_expanded = tf.expand_dims(attention_mask, -1)
            input_mask_expanded = tf.broadcast_to(input_mask_expanded, tf.shape(token_embeddings))

            sum_embeddings = tf.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = tf.reduce_sum(input_mask_expanded, axis=1)
            sum_mask = tf.clip_by_value(sum_mask, 1e-9, tf.float32.max)

            embeddings = sum_embeddings / sum_mask

            # Нормализация L2
            embeddings = tf.math.l2_normalize(embeddings, axis=1)

            return embeddings.numpy()[0]

    # ОСТАЛЬНЫЕ МЕТОДЫ БЕЗ ИЗМЕНЕНИЙ:
    # search(), add_documents(), _get_bm25_module() и т.д.
    # Они используют self._encode_text() который теперь работает с обоими бэкендами

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Каскадный поиск: BM25 → E5"""
        from src.core.rag import rag_engine

        # 1. Получаем BM25 модуль
        bm25_module = rag_engine.manager.search_modules.get(self.bm25_module_name)
        if not bm25_module:
            logger.error(f"BM25 модуль '{self.bm25_module_name}' не найден")
            return []

        # 2. BM25: быстрый поиск кандидатов
        bm25_results = bm25_module.search(query, top_k=self.top_k_candidates)

        if not bm25_results:
            return []

        logger.info(
            f"📊 BM25 нашел {len(bm25_results)} кандидатов, {('PyTorch' if TORCH_AVAILABLE else 'TensorFlow')} переранжирует..."
        )

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
                    "backend": "pytorch" if TORCH_AVAILABLE else "tensorflow",
                }
            )

        # 5. Сортировка и нормализация
        reranked.sort(key=lambda x: x["score"], reverse=True)

        if reranked:
            scores = [r["score"] for r in reranked]
            max_score = max(scores) if max(scores) > 0 else 1.0
            for r in reranked:
                r["score"] = r["score"] / max_score

        logger.info(
            f"✅ {('PyTorch' if TORCH_AVAILABLE else 'TensorFlow')} вернул {len(reranked[:top_k])} результатов"
        )

        return reranked[:top_k]

    def clear(self):
        self.doc_embeddings.clear()
        return {"status": "cleared", "name": self.name}

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "e5",
            "name": self.name,
            "backend": "pytorch" if TORCH_AVAILABLE else "tensorflow",
            "model_path": self.model_path,
            "bm25_source": self.bm25_module_name,
            "device": self.device,
            "embeddings_cached": len(self.doc_embeddings),
        }
