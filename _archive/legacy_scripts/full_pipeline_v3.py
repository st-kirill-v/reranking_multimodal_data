"""
RAG Evaluation Pipeline v3 — БЕЗ Markdown таблиц (только страницы PNG)

Ключевые изменения относительно v2:
  1. parse_answer()       — обрезает "internal thought:" из ответа VLM
  2. normalize_answer()   — нормализует ТОЛЬКО числовые ответы
  3. sanitize_generated() — обрезает хвост few-shot промпта из ответа VLM
  4. domain_from_folder() — домен из doc_report.json
  5. CONTEXT_LIMITS       — лимиты страниц по доменам
  6. expand_with_neighbors() — добавление соседних страниц
  7. search_by_figure_number() — поиск по номеру Figure/Table
"""

import torch
from transformers import AutoModel, AutoProcessor, AutoModelForSequenceClassification
from pathlib import Path
import faiss
import json
import numpy as np
from PIL import Image
import time
import sys
import os
import pickle
import re
import signal

FOLDER_DOMAIN_CACHE = {}


def load_domains_from_reports():
    """Загружает домены всех документов из doc_report.json"""
    global FOLDER_DOMAIN_CACHE
    for folder in data_path.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        report_path = folder / "extracted" / "doc_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                domain = report.get("domain", "academic")
                FOLDER_DOMAIN_CACHE[folder.name] = domain
    print(f"Loaded domains for {len(FOLDER_DOMAIN_CACHE)} documents")


def domain_from_folder(folder_name: str) -> str:
    """Возвращает домен документа по имени папки"""
    return FOLDER_DOMAIN_CACHE.get(folder_name, "academic")


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.modules.bm25_module import BM25Module
from src.core.generators.qwen_vl_generator import create_table_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_llm = "cuda"
device_rerank = "cuda"
device_embed = "cpu"

data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"
cache_dir = project_root / "cache" / "modules"
docling_dir = project_root / "data" / "docling_markdown_fast"


# ============================================================
# ТАЙМАУТ
# ============================================================


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Generation took too long")


# ============================================================
# КЛЮЧЕВЫЕ СЛОВА: ТИП ВОПРОСА
# ============================================================

TABLE_KEYWORDS = [
    "table",
    "row",
    "column",
    "cell",
    "value",
    "number",
    "percentage",
    "percent",
    "accuracy",
    "score",
    "f1",
    "bleu",
    "perplexity",
    "how many",
    "how much",
    "what is the total",
    "calculate",
    "sum",
    "difference",
    "average",
    "maximum",
    "minimum",
    "highest",
    "lowest",
    "increase",
    "decrease",
    "change",
    "growth",
    "compared to",
    "versus",
    "vs",
    "dataset",
    "result",
    "performance",
    "metric",
    "precision",
    "recall",
    "benchmark",
]

IMAGE_KEYWORDS = [
    "image",
    "picture",
    "photo",
    "graph",
    "plot",
    "chart",
    "scatter",
    "bar",
    "line",
    "pie",
    "histogram",
    "heatmap",
    "visual",
    "visualization",
    "screenshot",
    "snapshot",
    "drawing",
    "sketch",
    "figure",
    "illustration",
    "shown in",
    "depicted in",
]

FORMULA_KEYWORDS = [
    "formula",
    "equation",
    "mathematical",
    "expression",
    "computation",
    "parameter",
    "coefficient",
    "function",
    "f(x)",
    "loss function",
    "gradient",
    "derivative",
    "algorithm",
]


def determine_question_type(query: str) -> str:
    """Определяет тип вопроса: table | image | formula | text"""
    q = query.lower()
    table_score = sum(1 for kw in TABLE_KEYWORDS if kw in q)
    image_score = sum(1 for kw in IMAGE_KEYWORDS if kw in q)
    formula_score = sum(1 for kw in FORMULA_KEYWORDS if kw in q)
    scores = {"table": table_score, "image": image_score, "formula": formula_score}
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "text"
    return best_type


# ============================================================
# ПОСТПРОЦЕССИНГ ОТВЕТОВ
# ============================================================

GENERATION_CUTOFF_PATTERNS = [
    "Now answer the question",
    "EXAMPLES:",
    "Example 1",
    "Example 2",
    "Example 3",
    "RULES:",
    "ANALYSIS STEP:",
    "You are a precise RAG",
]


def sanitize_generated(text: str) -> str:
    if not text:
        return text
    for pattern in GENERATION_CUTOFF_PATTERNS:
        idx = text.find(pattern)
        if idx > 20:
            text = text[:idx].strip()
    return text


def parse_answer(raw_output: str) -> str:
    if not raw_output:
        return "NOT FOUND"
    text = re.sub(r"^assistant\s*\n", "", raw_output, flags=re.IGNORECASE).strip()
    text = sanitize_generated(text)
    lower = text.lower()
    if "internal thought:" in lower:
        idx = lower.find("internal thought:")
        after_thought = text[idx:]
        lines = [l.strip() for l in after_thought.split("\n") if l.strip()]
        content_lines = [l for l in lines[1:] if not l.lower().startswith("internal thought")]
        if content_lines:
            return content_lines[-1]
    return text.strip() if text.strip() else "NOT FOUND"


def normalize_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# ДИНАМИЧЕСКИЕ ЛИМИТЫ КОНТЕКСТА (только страницы PNG)
# ============================================================

# Только страницы PNG (максимум)
CONTEXT_LIMITS = {
    "financial": {"max_pages": 4},
    "academic": {"max_pages": 3},
    "government": {"max_pages": 3},
    "legal": {"max_pages": 3},
    "news": {"max_pages": 2},
}

MAX_PAGES_HARD_LIMIT = 6


def get_dynamic_limits(domain: str) -> dict:
    """Возвращает лимиты страниц для домена"""
    return CONTEXT_LIMITS.get(domain, {"max_pages": 3})


# ============================================================
# ПОИСК ПО НОМЕРУ ФИГУРЫ/ТАБЛИЦЫ
# ============================================================


def extract_figure_number(query: str):
    patterns = [
        (r"[Ff]igure\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)", "figure"),
        (r"[Ff]ig\.?\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)", "figure"),
        (r"[Tt]able\s+(\d+(?:\.\d+)?)", "table"),
    ]
    for pattern, fig_type in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1), fig_type
    return None, None


def search_by_figure_number(query: str, bm25_module, max_results: int = 5) -> list:
    fig_num, fig_type = extract_figure_number(query)
    if not fig_num:
        return []
    results = []
    seen = set()
    for term in [f"{fig_type.capitalize()} {fig_num}", f"{fig_type} {fig_num}"]:
        bm25_results = bm25_module.search(term, top_k=15)
        for r in bm25_results:
            doc_id = r.get("id", "")
            if "_" not in doc_id:
                continue
            folder, page_str = doc_id.split("_", 1)
            try:
                page_num = int(page_str)
            except ValueError:
                continue
            page_path = data_path / folder / "extracted" / "pages" / f"page_{page_num}.png"
            if not page_path.exists():
                continue
            key = f"{folder}_{page_num}"
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "folder": folder,
                    "page": page_num,
                    "score": 999.0,
                    "path": str(page_path),
                    "type": "page",
                }
            )
    return results[:max_results]


# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

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

print("Loading page index...")
page_index = faiss.read_index(str(index_dir / "pages_qwen3.index"))
with open(index_dir / "metadata_qwen3.json", "r", encoding="utf-8") as f:
    page_metadata = json.load(f)

print("Loading text index...")
text_index = faiss.read_index(str(project_root / "index_text_full" / "index.faiss"))
with open(project_root / "index_text_full" / "metadata.pkl", "rb") as f:
    text_metadata = pickle.load(f)

print("Loading image index...")
image_index = faiss.read_index(str(project_root / "index_images_full" / "index.faiss"))
with open(project_root / "index_images_full" / "metadata.pkl", "rb") as f:
    image_metadata = pickle.load(f)

print("Loading Nemotron Rerank...")
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

print("Loading BM25...")
bm25 = BM25Module(name="bm25", language="multilingual", k1=2.5, b=0.9)
bm25_path = cache_dir / "bm25"
if bm25_path.exists():
    bm25.load(str(cache_dir))
else:
    documents, doc_ids = [], []
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

print("Loading Qwen3-VL generator...")
qwen_generator = create_table_generator(device=device_llm)

print("Loading document domains...")
load_domains_from_reports()


# ============================================================
# ПОИСКОВЫЕ ФУНКЦИИ
# ============================================================


def encode_query(query: str) -> np.ndarray:
    with torch.no_grad():
        inputs = embed_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device_embed) for k, v in inputs.items()}
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def search_pages(query: str, emb: np.ndarray, top_k: int = 30) -> list:
    bm25_results = bm25.search(query, top_k=400)
    scores, indices = page_index.search(emb, 400)
    rrf_k = 30
    weight_bm25 = 2.0
    weight_embed = 1.0
    combined = {}
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        meta = page_metadata[idx]
        doc_id = f"{meta['folder']}_{meta['page']}"
        combined[doc_id] = {
            "score": weight_embed / (rank + rrf_k),
            "folder": meta["folder"],
            "page": meta["page"],
            "path": meta["path"],
            "type": "page",
        }
    for rank, r in enumerate(bm25_results):
        doc_id = r["id"]
        if doc_id in combined:
            combined[doc_id]["score"] += weight_bm25 / (rank + rrf_k)
        else:
            if "_" not in doc_id:
                continue
            folder, page = doc_id.split("_", 1)
            combined[doc_id] = {
                "score": weight_bm25 / (rank + rrf_k),
                "folder": folder,
                "page": int(page),
                "path": str(data_path / folder / "extracted" / "pages" / f"page_{int(page)}.png"),
                "type": "page",
            }
    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def search_text(emb: np.ndarray, top_k: int = 50) -> list:
    scores, indices = text_index.search(emb, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        meta = text_metadata[idx]
        folder_num = meta["folder"]
        md_file = docling_dir / str(folder_num) / "content.md"
        text_content = ""
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                text_content = f.read()
        results.append(
            {
                "rank": rank,
                "score": 1.0 / (30 + rank),
                "folder": folder_num,
                "type": "text",
                "metadata": meta,
                "text": text_content,
            }
        )
    return results


def search_images(emb: np.ndarray, top_k: int = 50) -> list:
    scores, indices = image_index.search(emb, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        meta = image_metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": 1.0 / (30 + rank),
                "folder": meta["folder"],
                "type": "image",
                "path": meta["path"],
                "metadata": meta,
            }
        )
    return results


def expand_with_neighbors(candidates: list, max_neighbors_per_page: int = 1) -> list:
    expanded = []
    for cand in candidates:
        expanded.append(cand)
        if cand.get("type") != "page":
            continue
        for delta in range(-max_neighbors_per_page, max_neighbors_per_page + 1):
            if delta == 0:
                continue
            neighbor_page = cand["page"] + delta
            if neighbor_page < 1:
                continue
            neighbor_path = Path(cand["path"]).parent / f"page_{neighbor_page}.png"
            if not neighbor_path.exists():
                continue
            neighbor = cand.copy()
            neighbor["page"] = neighbor_page
            neighbor["path"] = str(neighbor_path)
            neighbor["score"] = cand.get("score", 1.0) * 0.8
            expanded.append(neighbor)
    seen = set()
    unique = []
    for cand in expanded:
        key = f"{cand['folder']}_{cand.get('page', 0)}_{cand['type']}"
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def rerank_candidates(query: str, candidates: list, model, processor, device: str) -> list:
    if not candidates:
        return []
    reranked = []
    batch_size = 4
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        examples = []
        valid_batch = []
        for cand in batch:
            if cand["type"] in ("page", "image"):
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    examples.append({"question": query, "doc_text": "", "doc_image": img})
                    valid_batch.append(cand)
                except Exception:
                    continue
            elif cand["type"] == "text":
                text_content = cand.get("text", "")
                if not text_content:
                    continue
                examples.append(
                    {"question": query, "doc_text": text_content[:2000], "doc_image": None}
                )
                valid_batch.append(cand)
        if not examples:
            continue
        batch_dict = processor.process_queries_documents_crossencoder(examples)
        batch_dict = {k: v.to(device) for k, v in batch_dict.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            logits = model(**batch_dict, return_dict=True).logits
        for j, logit in enumerate(logits):
            if j < len(valid_batch):
                valid_batch[j]["rerank_score"] = float(torch.sigmoid(logit).item())
                reranked.append(valid_batch[j])
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


# ============================================================
# ГЛАВНЫЙ ПАЙПЛАЙН (ТОЛЬКО СТРАНИЦЫ PNG)
# ============================================================


def full_pipeline_v3(query: str, timeout_seconds: int = 180):
    start_time = time.time()

    print(f"\n[Q] {query[:100]}...")

    # Тип вопроса
    q_type = determine_question_type(query)
    print(f"[1] Type: {q_type}")

    # Поиск по Figure/Table
    figure_pages = search_by_figure_number(query, bm25)
    print(f"[2] Figure/table pages found: {len(figure_pages)}")

    # Кодирование запроса
    query_emb = encode_query(query)

    # Поиск страниц
    page_results = search_pages(query, query_emb, top_k=30)

    if figure_pages:
        existing_keys = {f"{r['folder']}_{r['page']}" for r in page_results}
        for fp in figure_pages:
            key = f"{fp['folder']}_{fp['page']}"
            if key not in existing_keys:
                page_results.insert(0, fp)

    # Определение домена
    domain = "academic"
    if page_results:
        folder_name = str(page_results[0].get("folder", ""))
        domain = domain_from_folder(folder_name)
    print(f"[3] Domain: {domain}")

    # Поиск текста и изображений (нужны для реранкинга, но не для контекста)
    text_results = search_text(query_emb, top_k=50)
    image_results = search_images(query_emb, top_k=50)

    # Реранкинг
    page_reranked = rerank_candidates(
        query, page_results, rerank_model, rerank_processor, device_rerank
    )
    text_reranked = rerank_candidates(
        query, text_results, rerank_model, rerank_processor, device_rerank
    )
    image_reranked = rerank_candidates(
        query, image_results, rerank_model, rerank_processor, device_rerank
    )
    print(
        f"[4] Reranked: {len(page_reranked)} pages, {len(text_reranked)} texts, {len(image_reranked)} images"
    )

    # Лимиты страниц по домену
    limits = get_dynamic_limits(domain)
    print(f"[5] Limits: pages={limits['max_pages']} (domain={domain})")

    # Отбор страниц и расширение соседями
    top_pages = page_reranked[: limits["max_pages"]]
    expanded_pages = expand_with_neighbors(top_pages, max_neighbors_per_page=1)
    expanded_pages = expanded_pages[:MAX_PAGES_HARD_LIMIT]

    # Сборка контекста — ТОЛЬКО PNG страницы
    context_images = []
    for cand in expanded_pages:
        try:
            img = Image.open(cand["path"]).convert("RGB")
            context_images.append(img)
        except Exception:
            continue

    print(f"[6] Context: {len(context_images)} images (PNG pages)")

    if not context_images:
        return "NOT FOUND", "not found", time.time() - start_time

    # Генерация
    print(f"[7] Generating (timeout={timeout_seconds}s)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        raw_answer = qwen_generator.generate_answer(query, context_images)
        print("[7] Generation completed")
    except TimeoutError:
        raw_answer = "TIMEOUT"
        print(f"[7] TIMEOUT after {timeout_seconds}s")
    finally:
        signal.alarm(0)

    # Постпроцессинг
    answer = parse_answer(raw_answer)
    answer_normalized = normalize_answer(answer)

    total_time = time.time() - start_time
    print(f"[8] Time: {total_time:.2f}s")
    print(f"[8] Raw: {raw_answer[:120]}")
    print(f"[8] Processed: {answer[:120]}")

    return answer, answer_normalized, total_time
