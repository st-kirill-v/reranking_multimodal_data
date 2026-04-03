from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import time
import json
from pathlib import Path
from src.core.metrics.search_metrics import MetricsReporter, SearchMetrics
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.queries.train_queries import get_train_relevant_ids
from datetime import datetime

from src.core.rag import rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    cache_dir = "cache/modules"

    try:
        # 1. Пытаемся ЗАГРУЗИТЬ существующий индекс
        bm25 = rag_engine.manager.search_modules.get("bm25")
        if bm25 and os.path.exists(os.path.join(cache_dir, "bm25")):
            print("Загружаю сохраненный индекс BM25...")
            result = bm25.load(cache_dir)  # ← load() а не load_state()!
            print(f"Результат загрузки: {result}")

            # Принудительно проверяем
            if hasattr(bm25, "is_fitted"):
                print(f"BM25 is_fitted: {bm25.is_fitted}")
            if hasattr(bm25, "get_document_count"):
                doc_count = bm25.get_document_count()
                print(f"Документов в BM25: {doc_count}")

        # 2. Если все еще 0 документов - строим индекс
        if bm25:
            if hasattr(bm25, "get_document_count"):
                doc_count = bm25.get_document_count()
            else:
                doc_count = len(bm25.documents) if bm25.documents else 0

            if doc_count == 0:
                print("Документов нет, строю индекс...")
                rag_engine.build_index()
            else:
                print(f"Документы загружены: {doc_count}")

    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        rag_engine.build_index()

    print(f"Система готова. Модулей: {len(rag_engine.manager.active_searchers)}")
    yield

    # Сохраняем при выключении
    try:
        bm25 = rag_engine.manager.search_modules.get("bm25")
        if bm25 and hasattr(bm25, "save"):
            bm25.save(cache_dir)
            print("Индекс BM25 сохранен")
    except Exception as e:
        print(f"Ошибка сохранения: {e}")


app = FastAPI(
    title="Modular RAG",
    description="Модульный RAG с поддержкой BM25, E5, CLIP и мультимодального поиска",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_reporter = MetricsReporter()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class AddDocumentsRequest(BaseModel):
    documents: List[str]
    ids: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    strategy: Optional[str] = "auto"


class MultimodalSearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 10
    use_rerank: Optional[bool] = True


class MultimodalSearchResult(BaseModel):
    id: str
    folder: str
    page: int
    score: float
    image_path: Optional[str] = None
    text_preview: Optional[str] = None


class MultimodalSearchResponse(BaseModel):
    query: str
    results: List[MultimodalSearchResult]
    search_time: float
    total_found: int
    rerank_used: bool


class AddModuleRequest(BaseModel):
    type: str
    name: Optional[str] = None
    config: Optional[Dict] = {}


class ExplainRequest(BaseModel):
    query: str
    document_type: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Modular RAG is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/info")
async def get_info():
    return rag_engine.get_info()


@app.post("/api/query")
async def api_query(request: QueryRequest):
    start_time = time.time()

    try:
        result = rag_engine.generate_answer(request.query, top_k=request.top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        # Собираем ID найденных документов
        retrieved_ids = []
        for i, source in enumerate(result.get("sources", [])):
            if isinstance(source, dict):
                doc_id = source.get("id", f"doc_{i}")
            else:
                doc_id = f"doc_{i}"
            retrieved_ids.append(doc_id)

        relevant_ids = get_train_relevant_ids(request.query)

        # Сохраняем метрики
        metrics_reporter.add_query_result(
            query_id=f"query_{int(time.time())}",
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            relevance_scores={doc_id: 1.0 for doc_id in relevant_ids},
            latency_ms=elapsed_ms,
        )

        return {
            "success": True,
            "query": result["query"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "total_found": result.get("total_found", 0),
            "latency_ms": elapsed_ms,
            "has_ground_truth": len(relevant_ids) > 0,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/search/multimodal", response_model=MultimodalSearchResponse)
async def multimodal_search(request: MultimodalSearchRequest):
    """
    Мультимодальный поиск по страницам документов (PNG)
    """
    try:
        start_time = time.time()

        # TODO: Здесь будет вызов реального мультимодального поиска из rag_engine
        # from src.core.multimodal_search import search_pages
        # results = await search_pages(request.query, request.n_results, request.use_rerank)

        # Пока заглушка для тестирования
        time.sleep(0.3)

        # Тестовые результаты
        test_results = [
            MultimodalSearchResult(
                id="0_3",
                folder="0",
                page=3,
                score=0.95,
                image_path="/data/datasets/docbench/0/extracted/pages/page_3.png",
                text_preview="Table 3: Language Model Perplexity",
            ),
            MultimodalSearchResult(
                id="0_7",
                folder="0",
                page=7,
                score=0.87,
                image_path="/data/datasets/docbench/0/extracted/pages/page_7.png",
                text_preview="Table 4: Fact Completion Results",
            ),
            MultimodalSearchResult(
                id="0_1",
                folder="0",
                page=1,
                score=0.76,
                image_path="/data/datasets/docbench/0/extracted/pages/page_1.png",
                text_preview="Barack's Wife Hillary: Using Knowledge Graphs...",
            ),
            MultimodalSearchResult(
                id="0_5",
                folder="0",
                page=5,
                score=0.68,
                image_path="/data/datasets/docbench/0/extracted/pages/page_5.png",
                text_preview="Linked WikiText-2 Corpus Statistics",
            ),
            MultimodalSearchResult(
                id="0_8",
                folder="0",
                page=8,
                score=0.59,
                image_path="/data/datasets/docbench/0/extracted/pages/page_8.png",
                text_preview="Completion Examples",
            ),
        ]

        # Обрезаем до запрошенного количества
        results = test_results[: request.n_results]

        search_time = time.time() - start_time

        return MultimodalSearchResponse(
            query=request.query,
            results=results,
            search_time=search_time,
            total_found=len(test_results),
            rerank_used=request.use_rerank,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/multimodal/health")
async def multimodal_health():
    """Проверка статуса мультимодального поиска"""
    # Проверяем наличие индекса
    index_path = Path("index/pages.index")
    metadata_path = Path("index/metadata.json")

    return {
        "status": "ok",
        "index_exists": index_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "model_embed": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "model_rerank": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    }


@app.get("/search/multimodal/test-queries")
async def get_test_queries(limit: int = 10):
    """Получить список тестовых мультимодальных запросов"""
    test_path = Path("data/test_queries.json")
    if not test_path.exists():
        return {"queries": []}

    with open(test_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    return {"queries": queries[:limit]}


@app.post("/documents")
async def add_documents(request: AddDocumentsRequest):
    try:
        result = rag_engine.add_documents(request.documents, request.ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = rag_engine.search(request.query, request.n_results, request.strategy)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    try:
        result = rag_engine.clear_documents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/modules")
async def add_module(request: AddModuleRequest):
    try:
        name = request.name or f"{request.type}_module"
        result = rag_engine.add_search_module(module_type=request.type, name=name, **request.config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/modules/{name}")
async def remove_module(name: str):
    try:
        result = rag_engine.remove_search_module(name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules")
async def list_modules():
    return rag_engine.list_modules()


@app.post("/load-squad-dataset")
async def load_squad_dataset(limit: Optional[int] = None):
    try:
        from src.pipeline.dataset_loader import load_squad_v2_local

        data = load_squad_v2_local()

        # Если limit указан - грузим часть, если нет - все
        if limit is not None:
            test_documents = data["documents"][:limit]
            test_ids = data["doc_ids"][:limit]
            print(f"Загружаю {limit} документов...")
        else:
            test_documents = data["documents"]
            test_ids = data["doc_ids"]
            print(f"Загружаю ВСЕ {len(test_documents)} документов...")

        # Очищаем старые документы перед загрузкой новых
        try:
            rag_engine.clear_documents()
            print("🧹 Старые документы очищены")
        except:
            pass

        # Загружаем новые
        result = rag_engine.add_documents(documents=test_documents, ids=test_ids)

        # Сохраняем состояние
        bm25 = rag_engine.manager.search_modules.get("bm25")
        if bm25 and hasattr(bm25, "save_state"):
            bm25.save_state()
            print("💾 Состояние сохранено")

        return {
            "status": "loaded",
            "documents_added": len(test_documents),
            "total_available": len(data["documents"]),
            "message": f"Добавлено {len(test_documents)} документов",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-router")
async def explain_router(request: ExplainRequest):
    try:
        router = rag_engine.manager.routers.get("smart")

        if not router:
            raise HTTPException(status_code=400, detail="SmartRouter не активирован")

        explanation = router.explain(request.query, request.document_type)
        explanation["timestamp"] = datetime.now().isoformat()

        return {"status": "success", "router": "smart", "explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-index")
async def rebuild_index():
    try:
        result = rag_engine.build_index()
        return {"status": "success", "message": "Index rebuilt", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index-status")
async def index_status():
    try:
        modules_info = []

        if rag_engine.manager.active_searchers and isinstance(
            rag_engine.manager.active_searchers[0], str
        ):
            for i, module_name in enumerate(rag_engine.manager.active_searchers):
                if module_name in rag_engine.manager.search_modules:
                    module = rag_engine.manager.search_modules[module_name]
                    module_info = {
                        "index": i,
                        "name": module_name,
                        "type": type(module).__name__,
                        "has_is_fitted": hasattr(module, "is_fitted"),
                        "has_documents": hasattr(module, "documents"),
                    }

                    if hasattr(module, "is_fitted"):
                        module_info["is_fitted"] = module.is_fitted
                    if hasattr(module, "documents"):
                        module_info["document_count"] = len(module.documents)
                    if hasattr(module, "get_document_count"):
                        module_info["document_count"] = module.get_document_count()

                    modules_info.append(module_info)
                else:
                    modules_info.append(
                        {
                            "index": i,
                            "name": module_name,
                            "type": "not_found",
                            "error": f"Module '{module_name}' not found in search_modules",
                        }
                    )
        else:
            for i, module in enumerate(rag_engine.manager.active_searchers):
                module_name = getattr(module, "name", f"module_{i}")
                module_info = {
                    "index": i,
                    "name": module_name,
                    "type": type(module).__name__,
                    "has_is_fitted": hasattr(module, "is_fitted"),
                    "has_documents": hasattr(module, "documents"),
                }

                if hasattr(module, "is_fitted"):
                    module_info["is_fitted"] = module.is_fitted
                if hasattr(module, "documents"):
                    module_info["document_count"] = len(module.documents)
                if hasattr(module, "get_document_count"):
                    module_info["document_count"] = module.get_document_count()

                modules_info.append(module_info)

        return {
            "active_modules_count": len(rag_engine.manager.active_searchers),
            "modules": modules_info,
            "search_modules_available": list(rag_engine.manager.search_modules.keys()),
            "active_searchers_raw": rag_engine.manager.active_searchers[:3],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Получить сводку метрик производительности."""
    return rag_engine.get_metrics_summary()


@app.get("/metrics/reset")
async def reset_metrics():
    """Сбросить собранные метрики."""
    return rag_engine.reset_metrics()


@app.get("/metrics/print")
async def print_metrics():
    """Вывести метрики в консоль и вернуть JSON."""
    try:
        summary = rag_engine.get_metrics_summary()

        if "error" in summary:
            return summary

        print("=== МЕТРИКИ ПОИСКА ===")
        if "latency_ms" in summary:
            lat = summary["latency_ms"]
            print(
                f"Латентность: средняя {lat['mean']:.1f}ms, p50 {lat['p50']:.1f}ms, p95 {lat['p95']:.1f}ms"
            )

        for key, value in summary.items():
            if key != "latency_ms" and isinstance(value, dict) and "mean" in value:
                print(f"{key}: {value['mean']:.3f} ± {value['std']:.3f}")

        return summary
    except Exception as e:
        return {"error": str(e)}


@app.get("/metrics/reranking")
async def get_reranking_metrics():
    """Получить метрики реранкинга"""
    summary = metrics_reporter.get_summary()
    if not summary:
        return {"message": "Нет данных метрик", "total_queries": 0}

    summary["total_queries"] = len(metrics_reporter.metrics_history)
    return summary


@app.get("/metrics/reranking/print")
async def print_reranking_metrics():
    """Вывести метрики в консоль"""
    print("МЕТРИКИ РЕРАНКИНГА")
    metrics_reporter.print_summary()
    return {"status": "printed"}


@app.post("/metrics/reranking/evaluate")
async def evaluate_reranking(request: dict):
    """Оценить конкретный запрос вручную"""
    try:
        retrieved_ids = request.get("retrieved_ids", [])
        relevant_ids = request.get("relevant_ids", [])

        metrics = {
            "recall@5": SearchMetrics.recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall@10": SearchMetrics.recall_at_k(retrieved_ids, relevant_ids, 10),
            "precision@5": SearchMetrics.precision_at_k(retrieved_ids, relevant_ids, 5),
            "mrr": SearchMetrics.mean_reciprocal_rank([retrieved_ids], [relevant_ids]),
        }

        return {"success": True, "metrics": metrics}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/metrics/reranking/reset")
async def reset_reranking_metrics():
    """Сбросить метрики"""
    metrics_reporter.metrics_history = []
    metrics_reporter.latencies_ms = []
    return {"status": "reset"}


@app.on_event("startup")
async def startup_event():
    """Выполняется после запуска сервера"""
    print("🚀 Запуск после старта сервера...")

    try:
        bm25 = rag_engine.manager.search_modules.get("bm25")
        if bm25:
            # Проверяем состояние
            if hasattr(bm25, "is_fitted"):
                print(f"BM25 is_fitted: {bm25.is_fitted}")

            if hasattr(bm25, "get_document_count"):
                doc_count = bm25.get_document_count()
                print(f"Документов в BM25: {doc_count}")
            elif hasattr(bm25, "documents"):
                doc_count = len(bm25.documents) if bm25.documents else 0
                print(f"Документов в BM25: {doc_count}")

            # Если есть документы но нет индекса - строим
            if hasattr(bm25, "documents") and bm25.documents and not bm25.is_fitted:
                print("Перестраиваю индекс BM25...")
                result = bm25.add_documents(bm25.documents)
                print(f"Результат: {result['status']}")

                # Сохраняем состояние
                if hasattr(bm25, "save_state"):
                    bm25.save_state()
                    print("Состояние сохранено")
    except Exception as e:
        print(f"Ошибка при старте: {e}")


if __name__ == "__main__":
    print("Modular RAG API запущен!")
    print("Адрес: http://127.0.0.1:8080")
    print("Документация: http://127.0.0.1:8080/docs")
    print("\nМультимодальные эндпоинты:")
    print("  POST /search/multimodal - мультимодальный поиск")
    print("  GET /search/multimodal/health - статус")
    print("  GET /search/multimodal/test-queries - тестовые запросы")

    uvicorn.run(app, host="127.0.0.1", port=8080)
