"""
RAG Evaluation Pipeline only pages.
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
# ПОСТПРОЦЕССИНГ ОТВЕТОВ
# ============================================================


def parse_answer(raw_output: str) -> str:
    """
    Убирает системные артефакты из ответа VLM:
      1. Префикс "assistant\\n" (добавляет Qwen)
      2. Блок "Internal Thought: ..." (из few-shot промпта)
      3. Хвост few-shot промпта (через sanitize_generated)

    Пример до:
        "Internal Thought: locate row Revenue -> value 242155\\n
         The total revenue was $242,155 million."
    Пример после:
        "The total revenue was $242,155 million."
    """
    if not raw_output:
        return "NOT FOUND"

    text = re.sub(r"^assistant\s*\n", "", raw_output, flags=re.IGNORECASE).strip()

    lower = text.lower()
    if "internal thought:" in lower:
        idx = lower.find("internal thought:")
        after_thought = text[idx:]
        lines = [l.strip() for l in after_thought.split("\n") if l.strip()]

        # Пропускаем строку с "internal thought:" и берём ПОСЛЕДНЮЮ строку
        # (финальный ответ всегда идёт после рассуждений)
        content_lines = [l for l in lines[1:] if not l.lower().startswith("internal thought")]
        if content_lines:
            # Последняя строка = финальный ответ
            return content_lines[-1]

    return text.strip() if text.strip() else "NOT FOUND"


def normalize_answer(text: str) -> str:
    """
    Нормализует ответ для вычисления метрик (F1 и Exact Match).

    ВАЖНО: применять к обоим — generated И expected — иначе нет смысла.
    Не меняет смысл ответа, только форматирование чисел.

    Что делает:
      - "25,000" → "25000"  (убирает числовые разделители тысяч)
      - "25.000" → "25.000" (НЕ трогает десятичные точки)
      - Приводит к нижнему регистру
      - Убирает лишние пробелы

    Что НЕ делает (принципиально):
      - НЕ срезает вводные фразы типа "The answer is..."
        Причина: expected ответы бывают двух типов:
          Тип А (число):    "65%", "44,723", "39.12"
          Тип Б (полное):   "BERT achieved the highest F1 with 93.42."
        Срезать вводные фразы у Типа Б сломает F1 — потеряем слова из expected.
        Для Типа А вводных фраз нет в ground truth, поэтому срезать нечего.

    Итог: единственное что реально помогает — нормализация чисел.
    Exact Match для Типа Б невозможен по определению (полные предложения
    не совпадут дословно). Оптимизировать стоит F1, а не EM.
    """
    # создаём t
    t = text.strip().lower()

    # 2. убрать разделители тысяч
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)

    # 3. нормализовать пробелы
    t = re.sub(r"\s+", " ", t).strip()

    return t


# ============================================================
# ПОИСК ПО НОМЕРУ ФИГУРЫ/ТАБЛИЦЫ
# ============================================================


def extract_figure_number(query: str):
    """Извлекает номер фигуры или таблицы из вопроса."""
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
    """
    Поиск страниц по номеру фигуры/таблицы.

    Fix v2: всегда возвращал 0 т.к. не проверял существование файла страницы.
    Теперь: явная проверка page_path.exists() перед добавлением кандидата.
    """
    fig_num, fig_type = extract_figure_number(query)
    if not fig_num:
        return []

    search_variants = [
        f"{fig_type.capitalize()} {fig_num}",
        f"{fig_type} {fig_num}",
        f"{fig_type.upper()} {fig_num}",
        f"{fig_type.capitalize()} {fig_num}.",  # с точкой
        f"{fig_type.capitalize()} {fig_num}:",  # с двоеточием
    ]
    results = []
    seen = set()

    for term in [f"{fig_type.capitalize()} {fig_num}", f"{fig_type} {fig_num}"]:
        bm25_results = bm25_module.search(term, top_k=15)
        for r in bm25_results:
            page_text = r.get("text", "").lower()
            # Мягкое matching — любой из вариантов
            matched = any(v.lower() in page_text for v in search_variants)
            if not matched:
                continue
            parts = r["id"].split("_")
            folder, page_str = parts[0], parts[1]
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
            parts = doc_id.split("_")
            if len(parts) >= 2:
                folder, page = parts[0], parts[1]
                combined[doc_id] = {
                    "score": weight_bm25 / (rank + rrf_k),
                    "folder": folder,
                    "page": int(page),
                    "path": str(
                        data_path / folder / "extracted" / "pages" / f"page_{int(page)}.png"
                    ),
                    "type": "page",
                }

    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def expand_with_neighbors(candidates: list, max_neighbors_per_page: int = 1) -> list:
    """
    Расширяет список кандидатов соседними страницами.

    Логика:
      Для каждой page-страницы добавляем ±N соседей (N = max_neighbors_per_page).
      Итого: len(candidates) × (1 + 2 × max_neighbors_per_page) страниц максимум.

    Финальный срез до MAX_PAGES_HARD_LIMIT делается в пайплайне явно,
    чтобы VLM не захлёбывался при 4-bit квантизации.

    Fix v2:
      - В v2 expand вызывался с max_pages=limits*2, но потом снаружи обрезался
        до limits — соседние страницы никогда не попадали в контекст.
      - Теперь expand возвращает ALL найденные страницы (оригинал + соседи),
        а финальный лимит контролируется снаружи через MAX_PAGES_HARD_LIMIT.
      - Добавлена проверка существования файлов соседних страниц.
      - Соседним страницам присваивается score × 0.8 чтобы они были ниже оригинала
        при повторной сортировке.
    """
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

    # Дедупликация по folder+page
    seen = set()
    unique = []
    for cand in expanded:
        key = f"{cand['folder']}_{cand.get('page', 0)}_{cand['type']}"
        if key not in seen:
            seen.add(key)
            unique.append(cand)

    return unique


def rerank_candidates(query: str, candidates: list, model, processor, device: str) -> list:
    """Реранкинг кандидатов через Nemotron cross-encoder."""
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
                    {
                        "question": query,
                        "doc_text": text_content[:2000],
                        "doc_image": None,
                    }
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
# ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================


def full_pipeline_only_pages(query: str, timeout_seconds: int = 180):
    start_time = time.time()

    print(f"\n[1/8] Question: {query[:80]}...")

    print(f"[2/7] Encoding query")
    # ── Encode ──────────────────────────
    query_emb = encode_query(query)

    # ── Search pages ────────────────────
    page_results = search_pages(query, query_emb, top_k=30)

    # ── Domain ──────────────────────────
    domain = "academic"
    if page_results:
        folder_name = page_results[0].get("folder", "")
        domain = domain_from_folder(folder_name)

    print(f"[3/8] Domain: {domain}")

    # ── Rerank pages ────────────────────
    page_reranked = rerank_candidates(
        query, page_results, rerank_model, rerank_processor, device_rerank
    )

    print(f"[4/8] Reranked pages: {len(page_reranked)}")

    # ── Берём топ страницы ──────────────
    top_pages = page_reranked[:5]

    # ── Expand neighbors ────────────────
    expanded_pages = expand_with_neighbors(top_pages, max_neighbors_per_page=1)

    # ограничение (очень важно)
    expanded_pages = expanded_pages[:8]

    expanded_pages = sorted(expanded_pages, key=lambda x: x.get("score", 0), reverse=True)[:6]

    print(f"[5/8] Final pages: {len(expanded_pages)}")

    print(f"[5.1] 📄 Pages sent to VLM:")
    for i, cand in enumerate(expanded_pages):
        folder = cand.get("folder", "?")
        page = cand.get("page", "?")
        print(f"        {i+1}. Document folder: {folder}, Page number: {page}")

    # ── Сборка контекста ────────────────
    context_images = []

    for cand in expanded_pages:
        try:
            img = Image.open(cand["path"]).convert("RGB")
            context_images.append(img)
        except Exception:
            continue

    if not context_images:
        return "NOT FOUND", "not found", time.time() - start_time

    print(f"[6/8] Images: {len(context_images)}")

    # ── Генерация ───────────────────────
    print(f"[7/8] Generating...")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        raw_answer = qwen_generator.generate_answer(query, context_images)
    except TimeoutError:
        raw_answer = "TIMEOUT"
    finally:
        signal.alarm(0)

    # ── Постпроцессинг ──────────────────
    answer = parse_answer(raw_answer)
    answer_normalized = normalize_answer(answer)

    total_time = time.time() - start_time

    print(f"[8/8] Done: {answer[:100]}")

    return answer, answer_normalized, total_time
