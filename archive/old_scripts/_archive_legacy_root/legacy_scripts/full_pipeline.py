import torch
from transformers import AutoModel, AutoProcessor, AutoModelForSequenceClassification
from pathlib import Path
from src.core.generators.qwen_vl_generator import create_table_generator
from src.core.modules.bm25_module import BM25Module
import faiss
import json
import numpy as np
from PIL import Image
import time
import sys
import os

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Используем GPU для реранкера и LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_llm = "cuda" if torch.cuda.is_available() else "cpu"
device_rerank = "cuda" if torch.cuda.is_available() else "cpu"
device_embed = "cpu"  # Эмбеддер на CPU для экономии VRAM

print(f"Using device for LLM: {device_llm}")
print(f"Using device for Rerank: {device_rerank}")
print(f"Using device for Embedding: {device_embed}")

# Пути
data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"
models_dir = project_root / "models" / "nemotron"
cache_dir = project_root / "cache" / "modules"

# Загрузка Qwen3-VL-Embedding на CPU
print("Loading Qwen3-VL-Embedding on CPU...")
embed_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_embed,
).eval()

embed_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True
)

# Загрузка индекса Qwen3-VL
print("Loading Qwen3-VL index...")
index = faiss.read_index(str(index_dir / "pages_qwen3.index"))
with open(index_dir / "metadata_qwen3.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Index loaded: {index.ntotal} vectors")

# Загрузка Nemotron Rerank на GPU
print("Loading Nemotron Rerank on GPU...")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/llama-nemotron-rerank-vl-1b-v2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_rerank,
).eval()
rerank_processor = AutoProcessor.from_pretrained(
    "nvidia/llama-nemotron-rerank-vl-1b-v2",
    trust_remote_code=True,
    max_input_tiles=6,
    use_thumbnail=True,
    rerank_max_length=2048,
)

# Загрузка BM25 модуля
print("Loading BM25...")
bm25 = BM25Module(name="bm25", language="multilingual", k1=2.5, b=0.9)
bm25_path = cache_dir / "bm25"
if bm25_path.exists():
    bm25.load(str(cache_dir))
else:
    print("Building BM25 from scratch...")
    documents = []
    doc_ids = []
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        text_file = folder / "extracted" / "pages_text.json"
        if text_file.exists():
            with open(text_file, "r", encoding="utf-8") as f:
                pages_text = json.load(f)
                for page in pages_text:
                    doc_id = f"{folder.name}_{page['page']}"
                    documents.append(page["text"])
                    doc_ids.append(doc_id)
    bm25.add_documents(documents, doc_ids)
    bm25.save(str(cache_dir))

# Загрузка Qwen3-VL генератора на GPU
print("Loading Qwen3-VL generator")
qwen_generator = create_table_generator(device=device_llm)


# Функция расширения соседними страницами
def expand_with_neighbors(candidates, max_pages=15):
    expanded = []
    for cand in candidates:
        expanded.append(cand)
        if cand["page"] > 1:
            prev_cand = cand.copy()
            prev_cand["page"] = cand["page"] - 1
            prev_cand["path"] = str(Path(cand["path"]).parent / f"page_{prev_cand['page']}.png")
            expanded.append(prev_cand)
        next_cand = cand.copy()
        next_cand["page"] = cand["page"] + 1
        next_cand["path"] = str(Path(cand["path"]).parent / f"page_{next_cand['page']}.png")
        expanded.append(next_cand)

    unique = {}
    for cand in expanded:
        key = f"{cand['folder']}_{cand['page']}"
        if key not in unique:
            unique[key] = cand
    return list(unique.values())[:max_pages]


# Гибридный поиск с Qwen3-VL-Embedding (эмбеддер на CPU)
def full_search(query, top_k_initial=400, top_k_rerank=150, final_k=30):
    start_time = time.time()

    # BM25 поиск
    bm25_results = bm25.search(query, top_k=top_k_initial)

    # Qwen3-VL-Embedding для текстового запроса на CPU
    with torch.no_grad():
        inputs = embed_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device_embed) for k, v in inputs.items()}

        outputs = embed_model(**inputs)

        # Mean pooling по всем токенам
        query_emb = outputs.last_hidden_state.mean(dim=1)

        # Нормализация
        query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)

    # Поиск в FAISS индексе на CPU
    scores, indices = index.search(query_emb.cpu().float().numpy(), top_k_initial)

    # RRF слияние результатов
    rrf_k = 30
    weight_bm25 = 2.0
    weight_embed = 1.0

    combined = {}

    # Добавляем результаты из векторного поиска
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        doc_id = f"{meta['folder']}_{meta['page']}"
        combined[doc_id] = {
            "score": weight_embed / (rank + rrf_k),
            "folder": meta["folder"],
            "page": meta["page"],
            "path": meta["path"],
        }

    # Добавляем результаты BM25
    for rank, r in enumerate(bm25_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["score"] += weight_bm25 / (rank + rrf_k)
        else:
            parts = doc_id.split("_")
            if len(parts) >= 2:
                folder, page = parts[0], parts[1]
                try:
                    page_int = int(page)
                    combined[doc_id] = {
                        "score": weight_bm25 / (rank + rrf_k),
                        "folder": folder,
                        "page": page_int,
                        "path": str(
                            data_path / folder / "extracted" / "pages" / f"page_{page_int}.png"
                        ),
                    }
                except:
                    pass

    # Сортируем и берём top_k_rerank
    candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k_rerank]

    # Реренкинг с Nemotron на GPU
    reranked = []
    batch_size = 4
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        examples = []
        for cand in batch:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                examples.append({"question": query, "doc_text": "", "doc_image": img})
            except:
                continue
        if not examples:
            continue
        batch_dict = rerank_processor.process_queries_documents_crossencoder(examples)
        batch_dict = {
            k: v.to(device_rerank) for k, v in batch_dict.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            logits = rerank_model(**batch_dict, return_dict=True).logits
        for j, logit in enumerate(logits):
            if j < len(batch):
                batch[j]["rerank_score"] = float(torch.sigmoid(logit).item())
                reranked.append(batch[j])

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    elapsed = time.time() - start_time
    return reranked[:final_k], elapsed


# Полный пайплайн
def full_pipeline(query, top_k_initial=400, top_k_rerank=150, final_k=30):
    start_time = time.time()

    candidates, search_time = full_search(query, top_k_initial, top_k_rerank, final_k)

    print(f"\nSearch time: {search_time:.2f}s")
    print(f"\nTop-10 after rerank:")
    for i, c in enumerate(candidates[:10]):
        print(f"  {i+1}. Doc {c['folder']}, Page {c['page']}, Score: {c['rerank_score']:.4f}")

    # Расширяем соседними страницами
    candidates_with_neighbors = expand_with_neighbors(candidates, max_pages=15)

    # Загружаем все страницы для LLM
    page_images = []
    for cand in candidates_with_neighbors[:15]:
        try:
            img = Image.open(cand["path"]).convert("RGB")
            page_images.append(img)
        except:
            continue

    if not page_images:
        return "NOT FOUND", time.time() - start_time

    print(f"\nLoaded {len(page_images)} pages for Qwen3-VL (including neighbors)")

    # Генерация ответа на GPU
    print(f"\nGenerating answer with Qwen3-VL...")
    answer = qwen_generator.generate_answer(query, page_images)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Answer: {answer}")

    return answer, total_time


# Тест
if __name__ == "__main__":
    test_query = "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?"
    print(f"\nTest query: {test_query}")

    answer, total_time = full_pipeline(test_query)

    print(f"\nFinal answer: {answer}")
