from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

from src.core.rag import rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        result = rag_engine.build_index()
        print(f"Индекс построен")
    except Exception as e:
        print(f"Предупреждение при построении индекса: {e}")
        print("Возможно индекс уже построен или нет документов")

    print(f"Система готова. Модулей: {len(rag_engine.manager.active_searchers)}")
    yield


app = FastAPI(
    title="Modular RAG",
    description="Модульный RAG с поддержкой BM25, E5, CLIP",
    version="2.0.0",
    lifespan=lifespan,
)


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
    try:
        result = rag_engine.generate_answer(request.query, top_k=request.top_k)
        return {
            "success": True,
            "query": result["query"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "total_found": result.get("total_found", 0),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
async def load_squad_dataset():
    try:
        from src.pipeline.dataset_loader import load_squad_v2_local

        data = load_squad_v2_local()
        test_documents = data["documents"][:1000]
        test_ids = data["doc_ids"][:1000]

        result = rag_engine.add_documents(documents=test_documents, ids=test_ids)

        return {
            "status": "loaded",
            "documents_added": len(test_documents),
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


print("Автоматически перестраиваю BM25...")
try:
    bm25 = rag_engine.manager.search_modules.get("bm25")
    if bm25 and hasattr(bm25, "documents") and bm25.documents and not bm25.is_fitted:
        print(f"Документов: {len(bm25.documents)}, is_fitted: {bm25.is_fitted}")
        result = bm25.add_documents(bm25.documents)
        print(f"Результат: {result['status']}")
except Exception as e:
    print(f"Ошибка перестройки BM25: {e}")

if __name__ == "__main__":
    print("Modular RAG API запущен")
    print("http://localhost:8000/docs")
    print("Для проверки индекса: GET /index-status")
    print("Для перестроения: POST /rebuild-index")
    uvicorn.run(app, host="0.0.0.0", port=8000)
