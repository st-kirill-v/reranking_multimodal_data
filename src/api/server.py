from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from contextlib import asynccontextmanager

# Добавляем корневую директорию проекта в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.core.rag import rag_engine


# Lifespan менеджер вместо on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup логика - загрузка индекса при запуске
    if rag_engine.load_index():
        print("Индекс загружен с диска")
    else:
        print("Индекс не найден, начинаем с чистого состояния")
    yield
    # Shutdown логика может быть добавлена здесь при необходимости


app = FastAPI(title="Text RAG", description="Текстовый RAG", version="1.0.0", lifespan=lifespan)


class AddDocumentsRequest(BaseModel):
    documents: List[str]
    ids: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5


@app.get("/")
async def root():
    return {"message": "Text RAG is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/info")
async def get_info():
    return rag_engine.get_info()


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
        results = rag_engine.search(request.query, request.n_results)
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


@app.post("/load-squad-dataset")
async def load_squad_dataset():
    """Загружает SQuAD 2.0 датасет в RAG систему"""
    try:
        # Импортируем здесь чтобы избежать циклических импортов
        from src.pipeline.dataset_loader import load_squad_v2_local

        # Загружаем данные
        data = load_squad_v2_local()

        # Добавляем в RAG (первые 1000 для теста)
        test_documents = data["documents"][:1000]
        test_ids = data["doc_ids"][:1000]

        result = rag_engine.add_documents(documents=test_documents, ids=test_ids)

        return {
            "status": "loaded",
            "documents_added": len(test_documents),
            "total_documents": len(rag_engine.documents),
            "message": f"Добавлено {len(test_documents)} документов из SQuAD 2.0",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
