"""
Главный RAG движок с модульной архитектурой и LLM генерацией.
"""

from src.core.module_manager import ModuleManager
from src.core.modules.bm25_module import BM25Module
from src.core.modules.fusion_modules import RRFusion
from src.core.modules.router_modules import DebugRouter
from src.core.metrics.search_metrics import MetricsReporter
from src.core.generators.yandex_gpt_generator import create_llm_generator
from src.core.metrics.search_metrics import MetricsReporter, SearchMetrics
from typing import List, Dict, Any, Optional
import torch
import time


class ModularRAG:
    def __init__(self, storage_path: str = "data/modules"):
        self.manager = ModuleManager(storage_path)
        self._index_built = False
        self.llm_generator = create_llm_generator("yandexgpt")
        self.metrics_reporter = MetricsReporter()

        try:
            self._init_default_modules()
            self.manager.load_all()
            print("RAG система инициализирована.")
        except Exception as e:
            print(f"Ошибка инициализации RAG: {e}")

    def get_metrics_summary(self):
        """Получить сводку метрик."""
        return self.metrics_reporter.get_summary()

    def print_metrics_summary(self):
        """Вывести метрики."""
        self.metrics_reporter.print_summary()

    def _init_default_modules(self):
        from src.core.modules.e5_module import E5Module

        bm25_module = BM25Module(name="bm25", language="multilingual")
        self.manager.register_search_module(bm25_module, activate=True)

        e5_module = E5Module(
            name="e5_reranker",
            model_path="./models/e5/e5-small-v2",
            bm25_module_name="bm25",
            top_k_candidates=100,
            device="cuda" if torch.cuda.is_available() else "cpu",
            manager=self.manager,
            storage_path=self.manager.storage_path,
        )
        self.manager.register_search_module(e5_module, activate=True)

        rrf = RRFusion()
        self.manager.register_fusion_module("rrf", rrf, activate=True)

        router = DebugRouter()
        self.manager.register_router("smart", router, activate=True)

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        try:
            result = self.manager.add_documents(documents, ids)
            print(f"Добавлено {len(documents)} документов.")
            return result
        except Exception as e:
            print(f"Ошибка добавления документов: {e}")
            return {"status": "error", "message": str(e)}

    def search(
        self, query: str, n_results: int = 5, strategy: str = "auto", relevant_ids: List[str] = None
    ) -> Dict:
        """Поиск с метриками и латентностью."""
        import time

        start_time = time.time()

        try:
            # Выполняем поиск
            result = self.manager.search(query, n_results, strategy)

            # Время выполнения
            latency_ms = (time.time() - start_time) * 1000

            # Форматируем результаты
            formatted_results = []
            for doc in result["results"]:
                formatted_results.append(
                    {
                        "id": doc.get("id"),
                        "content": doc.get("content", ""),
                        "score": doc.get("fusion_score", doc.get("score", 0.0)),
                        "module": doc.get("module", "unknown"),
                        "method": doc.get("method", "unknown"),  # Добавляем метод
                    }
                )

            # Нормализация scores
            if formatted_results and len(formatted_results) > 1:
                scores = [doc["score"] for doc in formatted_results]
                max_score = max(scores)
                min_score = min(scores)

                if max_score > min_score:
                    for doc in formatted_results:
                        orig_score = doc["score"]
                        norm_score = (orig_score - min_score) / (max_score - min_score)
                        doc["score"] = min(1.0, max(0.0, norm_score))

                formatted_results.sort(key=lambda x: x["score"], reverse=True)
                formatted_results = formatted_results[:n_results]

            # Собираем метрики если есть ground truth
            if relevant_ids and hasattr(self, "metrics_reporter"):
                retrieved_ids = [doc.get("id") for doc in formatted_results]

                # Создаем relevance_scores (1.0 для релевантных, 0.0 для остальных)
                relevance_scores = {}
                for doc in formatted_results:
                    doc_id = doc.get("id")
                    relevance_scores[doc_id] = 1.0 if doc_id in relevant_ids else 0.0

                self.metrics_reporter.add_query_result(
                    query_id=query,  # или можно использовать хэш запроса
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    relevance_scores=relevance_scores,
                    latency_ms=latency_ms,
                )

            print(
                f"Поиск выполнен. Найдено результатов: {len(formatted_results)}, время: {latency_ms:.1f}мс"
            )

            # Возвращаем результат с дополнительной информацией
            return {
                "query": query,
                "results": formatted_results,
                "strategy": result.get("strategy", strategy),
                "modules_used": result.get("modules_used", []),
                "latency_ms": round(latency_ms, 2),
                "normalized": True,
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"Ошибка поиска: {e}, время: {latency_ms:.1f}мс")
            return {
                "query": query,
                "results": [],
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
            }

    def generate_answer(self, query: str, top_k: int = 3, min_score: float = 0.1) -> Dict:
        try:
            start_time = time.time()

            search_results = self.search(query, n_results=top_k * 3)["results"]

            filtered_results = [doc for doc in search_results if doc.get("score", 0) >= min_score][
                :top_k
            ]

            formatted_sources = []
            for i, doc in enumerate(filtered_results, 1):
                content_preview = doc.get("content", "")[:150]
                if len(doc.get("content", "")) > 150:
                    content_preview += "..."

                formatted_sources.append(
                    {
                        "id": doc.get("id"),
                        "preview": content_preview,
                        "score": round(doc.get("score", 0), 3),
                        "module": doc.get("module", "unknown"),
                    }
                )

            retrieved_ids = []
            relevance_scores = {}

            for i, doc in enumerate(search_results[: top_k * 2]):
                doc_id = doc.get("id", f"doc_{i}")
                retrieved_ids.append(doc_id)
                relevance_scores[doc_id] = doc.get("score", 0)

            if not filtered_results:
                latency_ms = (time.time() - start_time) * 1000

                self.metrics_reporter.add_query_result(
                    query_id=query[:20] if query else "empty",
                    retrieved_ids=retrieved_ids,
                    relevant_ids=[],
                    relevance_scores=relevance_scores,
                    latency_ms=latency_ms,
                )

                return {
                    "query": query,
                    "answer": "No relevant information found.",
                    "sources": formatted_sources,  # Теперь будет пустой список
                    "total_found": len(search_results),
                    "metrics": {
                        "search_time_ms": round(latency_ms, 2),
                        "documents_found": len(search_results),
                    },
                }

            context_docs = []
            for doc in filtered_results:
                context_docs.append(
                    {"content": doc.get("content", ""), "score": doc.get("score", 0)}
                )

            answer = self.llm_generator.generate_answer(query, context_docs)
            latency_ms = (time.time() - start_time) * 1000

            relevant_ids = retrieved_ids[:2] if len(retrieved_ids) >= 2 else retrieved_ids

            self.metrics_reporter.add_query_result(
                query_id=query[:20] if query else "unknown",
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                relevance_scores=relevance_scores,
                latency_ms=latency_ms,
            )

            print(f"Ответ сгенерирован. Использовано источников: {len(filtered_results)}")

            result = {
                "query": query,
                "answer": answer,
                "sources": formatted_sources,  # Уже создан
                "total_found": len(search_results),
                "used_sources": len(filtered_results),
                "generator_info": self.llm_generator.get_info(),
                "metrics": {
                    "search_time_ms": round(latency_ms, 2),
                    "documents_found": len(search_results),
                    "relevant_found": len(filtered_results),
                },
            }

            return result

        except Exception as e:
            print(f"Ошибка генерации ответа: {e}")
            return {
                "query": query,
                "answer": f"Based on found information: {search_results[0].get('content', '')[:200] if search_results else 'No information'}",
                "sources": search_results[:top_k],
                "error": str(e),
            }

    def get_metrics_summary(self) -> Dict:
        return self.metrics_reporter.get_summary()

    def reset_metrics(self) -> Dict:
        self.metrics_reporter = MetricsReporter()
        return {"success": True, "message": "Метрики сброшены"}

    def get_info(self) -> Dict:
        info = self.manager.get_info()
        info["llm_generator"] = self.llm_generator.get_info()
        print("Информация о системе получена.")
        return info

    def clear_documents(self) -> Dict:
        try:
            for name, module in self.manager.search_modules.items():
                module.clear()

            self.manager.save_all()
            print("Все документы очищены.")
            return {"status": "cleared"}
        except Exception as e:
            print(f"Ошибка очистки документов: {e}")
            return {"status": "error", "message": str(e)}

    def build_index(self) -> Dict[str, Any]:
        try:
            results = {}

            for module_name in self.manager.active_searchers:
                if module_name not in self.manager.search_modules:
                    results[module_name] = {"status": "error", "message": "Module not found"}
                    continue

                module = self.manager.search_modules[module_name]

                try:
                    if hasattr(self.manager, "documents") and self.manager.documents:
                        documents = self.manager.documents
                    elif hasattr(module, "documents") and module.documents:
                        documents = module.documents
                    else:
                        results[module_name] = {"status": "error", "message": "No documents"}
                        continue

                    if hasattr(module, "build_index"):
                        result = module.build_index()
                        results[module_name] = result
                    elif hasattr(module, "fit"):
                        result = module.fit(documents)
                        results[module_name] = {
                            "status": "success",
                            "method": "fit",
                            "details": result,
                        }
                    else:
                        results[module_name] = {
                            "status": "skipped",
                            "message": "No indexing method",
                        }

                except Exception as e:
                    results[module_name] = {"status": "error", "message": str(e)}

            self._index_built = True
            self.manager.save_all()

            print(f"Индекс построен. Модулей обработано: {len(results)}")
            return {
                "status": "success",
                "details": {
                    "index_built": True,
                    "total_modules": len(results),
                    "results": results,
                },
            }
        except Exception as e:
            print(f"Ошибка построения индекса: {e}")
            return {"status": "error", "message": str(e)}

    def is_index_built(self) -> bool:
        for name in self.manager.active_searchers:
            if name in self.manager.search_modules:
                module = self.manager.search_modules[name]
                if hasattr(module, "is_fitted") and not module.is_fitted:
                    return False
        return True

    def get_document_count(self) -> int:
        total_count = 0
        for name, module in self.manager.search_modules.items():
            if hasattr(module, "get_document_count"):
                total_count += module.get_document_count()
            elif hasattr(module, "documents"):
                total_count += len(module.documents)

        return total_count

    def get_system_status(self) -> Dict:
        module_status = {}

        for name, module in self.manager.search_modules.items():
            status = {
                "active": name in self.manager.active_searchers,
                "type": module.__class__.__name__,
            }

            if hasattr(module, "documents") and module.documents is not None:
                status["doc_count"] = len(module.documents)
            elif hasattr(module, "get_document_count"):
                try:
                    status["doc_count"] = module.get_document_count()
                except:
                    status["doc_count"] = "unknown"
            else:
                status["doc_count"] = "unknown"

            if hasattr(module, "is_fitted"):
                status["index_ready"] = module.is_fitted
            else:
                status["index_ready"] = "N/A"

            module_status[name] = status

        return {
            "total_documents": self.get_document_count(),
            "index_built": self._index_built,
            "active_modules": list(self.manager.active_searchers),
            "modules": module_status,
            "llm_loaded": self.llm_generator is not None,
        }

    def add_search_module(self, module_type: str, name: str, **kwargs) -> Dict:
        try:
            if module_type == "e5":
                from src.core.modules.e5_module import E5Module

                module = E5Module(name=name, **kwargs)
                self.manager.register_search_module(module, activate=True)
                result = {"status": "added", "name": name, "type": "e5"}
            elif module_type == "clip":
                from src.core.modules.clip_module import CLIPModule

                module = CLIPModule(name=name, **kwargs)
                self.manager.register_search_module(module, activate=True)
                result = {"status": "added", "name": name, "type": module_type}
            elif module_type == "layoutlm":
                from src.core.modules.layoutlm_module import LayoutLMModule

                module = LayoutLMModule(name=name, **kwargs)
                self.manager.register_search_module(module, activate=True)
                result = {"status": "added", "name": name, "type": module_type}
            else:
                return {"status": "error", "message": f"Unknown module type: {module_type}"}

            print(f"Модуль {name} типа {module_type} добавлен.")
            return result
        except Exception as e:
            print(f"Ошибка добавления модуля: {e}")
            return {"status": "error", "message": str(e)}


rag_engine = ModularRAG()
