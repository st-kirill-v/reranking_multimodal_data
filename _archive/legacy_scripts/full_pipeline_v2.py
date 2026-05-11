"""
RAG Evaluation Pipeline v2 — исправленная версия после разбора вопросов.
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
    """
    Определяет тип вопроса: table | image | formula | text

    Логика:
      - Если есть пересечения с TABLE_KEYWORDS → table
      - Если есть пересечения с IMAGE_KEYWORDS → image
      - Если есть пересечения с FORMULA_KEYWORDS → formula
      - Иначе → text (дефолт)
    """
    q = query.lower()

    table_score = sum(1 for kw in TABLE_KEYWORDS if kw in q)
    image_score = sum(1 for kw in IMAGE_KEYWORDS if kw in q)
    formula_score = sum(1 for kw in FORMULA_KEYWORDS if kw in q)

    scores = {
        "table": table_score,
        "image": image_score,
        "formula": formula_score,
    }

    best_type = max(scores, key=scores.get)

    # если вообще ничего не найдено → text
    if scores[best_type] == 0:
        return "text"

    return best_type


# ============================================================
# КЛЮЧЕВЫЕ СЛОВА: ДОМЕН ДОКУМЕНТА (5 доменов DocBench)
# ============================================================

# Финансовые отчёты (10-K, annual reports, earnings)
FINANCIAL_KEYWORDS = {
    "revenue",
    "profit",
    "income",
    "assets",
    "debt",
    "earnings",
    "fiscal",
    "quarter",
    "shareholders",
    "dividends",
    "equity",
    "liabilities",
    "cash flow",
    "operating",
    "net income",
    "gross",
    "ebitda",
    "balance sheet",
    "annual report",
    "10-k",
    "consolidated",
    "segment",
    "capex",
    "expenditure",
    "sales",
    "margin",
    "amortization",
}

# Академические NLP/ML статьи
ACADEMIC_KEYWORDS = {
    "model",
    "bert",
    "training",
    "dataset",
    "corpus",
    "nlp",
    "accuracy",
    "f1",
    "bleu",
    "perplexity",
    "attention",
    "transformer",
    "embedding",
    "classification",
    "entity",
    "coreference",
    "summarization",
    "translation",
    "neural",
    "layer",
    "epoch",
    "baseline",
    "ablation",
}

# Правительственные документы
GOVERNMENT_KEYWORDS = {
    "policy",
    "regulation",
    "government",
    "federal",
    "agency",
    "department",
    "legislation",
    "act",
    "section",
    "statute",
    "public",
    "authority",
    "ministry",
    "cabinet",
    "parliament",
    "senate",
    "bill",
}

# Юридические документы
LEGAL_KEYWORDS = {
    "contract",
    "clause",
    "agreement",
    "liability",
    "plaintiff",
    "defendant",
    "court",
    "jurisdiction",
    "legal",
    "law",
    "attorney",
    "counsel",
    "jurisdiction",
    "indemnification",
    "warranty",
    "breach",
}

# Новостные статьи
NEWS_KEYWORDS = {
    "according to",
    "reported",
    "announced",
    "statement",
    "spokesperson",
    "interview",
    "journalist",
    "newspaper",
    "article",
    "press release",
    "editorial",
    "breaking",
    "sources said",
    "confirmed",
}

# Паттерны для обрезки хвоста few-shot промпта из ответа VLM
# Проблема была в том что генератор иногда не завершал ответ и "съезжал"
# обратно на текст few-shot примеров из промпта
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

# ============================================================
# ПОСТПРОЦЕССИНГ ОТВЕТОВ
# ============================================================


def sanitize_generated(text: str) -> str:
    """
    Обрезает хвост few-shot промпта из ответа VLM.

    Проблема (вопрос [194] в логе v2):
        Generated: "BMW achieved the highest performance score with 92%.
                    Now answer the question following the exact same format..."
    Причина: VLM не завершил генерацию и вернул хвост few-shot промпта.
    Это происходит внутри генератора, не в retrieval — поэтому sanitize_chunk()
    в pipeline не поможет. Нужно обрезать ответ здесь.

    Логика: если паттерн найден НЕ в самом начале (idx > 20 символов),
    берём только текст до него.
    """
    if not text:
        return text
    for pattern in GENERATION_CUTOFF_PATTERNS:
        idx = text.find(pattern)
        if idx > 20:  # паттерн не в начале — это хвост промпта
            text = text[:idx].strip()
    return text


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
    text = sanitize_generated(text)

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


def domain_classifier(query: str) -> str:
    """
    Определяет домен документа по тексту вопроса.
    5 доменов DocBench: financial | academic | government | legal | news

    Зачем нужно:
      Финансовые вопросы стабильно получали страницы из NLP статей в v2.
      Причина: BM25 по "net revenue 2020" матчит академические статьи где
      "operating" встречается в контексте "operating on the dataset".
      Реранкер не всегда исправляет это при жёстком лимите pages=1.

      domain_classifier позволяет:
        1. Выдать финансовым вопросам больше страниц (pages=3 vs pages=1)
        2. В будущем: фильтровать кандидатов по типу документа из metadata

    Метод: подсчёт пересечений слов вопроса со словарями доменов.
    Победитель — домен с наибольшим счётом. При ничьей → academic (дефолт).
    """
    q = query.lower()
    scores = {
        "financial": sum(1 for kw in FINANCIAL_KEYWORDS if kw in q),
        "academic": sum(1 for kw in ACADEMIC_KEYWORDS if kw in q),
        "government": sum(1 for kw in GOVERNMENT_KEYWORDS if kw in q),
        "legal": sum(1 for kw in LEGAL_KEYWORDS if kw in q),
        "news": sum(1 for kw in NEWS_KEYWORDS if kw in q),
    }
    best_domain = max(scores, key=scores.get)
    # Если все счётчики 0 — дефолт academic
    return best_domain if scores[best_domain] > 0 else "academic"


# ============================================================
# ДИНАМИЧЕСКИЕ ЛИМИТЫ КОНТЕКСТА
# ============================================================

# Явная таблица: домен × тип_вопроса → лимиты контекста
#
# Логика выбора значений:
#   financial: таблицы многостраничные → pages=3, tables=3
#   academic:  статьи компактные, вопросы про figures → pages=2 для image/formula
#   government/legal: длинные текстовые документы → больше страниц для text
#   news:      короткие документы → меньше страниц
#
# max_tables — сколько Markdown-таблиц извлекать из text chunks
# max_images — сколько отдельных изображений из image index
# max_pages  — сколько PNG страниц передавать в VLM

CONTEXT_LIMITS = {
    "financial": {
        "table": {"max_pages": 3, "max_tables": 3, "max_images": 0},
        "text": {"max_pages": 4, "max_tables": 1, "max_images": 0},
        "image": {"max_pages": 2, "max_tables": 0, "max_images": 3},
        "formula": {"max_pages": 2, "max_tables": 0, "max_images": 3},
    },
    "academic": {
        "table": {"max_pages": 2, "max_tables": 2, "max_images": 0},
        "text": {"max_pages": 5, "max_tables": 0, "max_images": 0},
        "image": {"max_pages": 1, "max_tables": 0, "max_images": 4},
        "formula": {"max_pages": 1, "max_tables": 0, "max_images": 4},
    },
    "government": {
        "table": {"max_pages": 2, "max_tables": 2, "max_images": 0},
        "text": {"max_pages": 5, "max_tables": 0, "max_images": 0},
        "image": {"max_pages": 2, "max_tables": 0, "max_images": 3},
        "formula": {"max_pages": 2, "max_tables": 0, "max_images": 3},
    },
    "legal": {
        "table": {"max_pages": 2, "max_tables": 2, "max_images": 0},
        "text": {"max_pages": 5, "max_tables": 0, "max_images": 0},
        "image": {"max_pages": 2, "max_tables": 0, "max_images": 2},
        "formula": {"max_pages": 2, "max_tables": 0, "max_images": 2},
    },
    "news": {
        "table": {"max_pages": 2, "max_tables": 2, "max_images": 0},
        "text": {"max_pages": 3, "max_tables": 0, "max_images": 0},
        "image": {"max_pages": 2, "max_tables": 0, "max_images": 3},
        "formula": {"max_pages": 1, "max_tables": 0, "max_images": 3},
    },
}

# Жёсткий потолок страниц передаваемых в VLM (включая соседей)
# Больше этого числа — VLM начинает "захлёбываться" при 4-bit
MAX_PAGES_HARD_LIMIT = 6


def get_dynamic_limits(q_type: str, domain: str) -> dict:
    """
    Возвращает лимиты контекста для данного типа вопроса и домена.
    Если домен неизвестен — fallback на academic.
    """
    domain_limits = CONTEXT_LIMITS.get(domain, CONTEXT_LIMITS["academic"])
    return domain_limits.get(q_type, domain_limits["text"])


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
# ИЗВЛЕЧЕНИЕ ТАБЛИЦ ИЗ MARKDOWN
# ============================================================


def extract_tables_from_markdown(markdown_text: str, max_chars: int = 800) -> list:
    """
    Извлекает Markdown-таблицы из текста.
    max_chars=800 (в v2 было 500 — финансовые таблицы обрезались слишком рано).
    """
    table_pattern = r"(\|.+\|[\s]*[\|\-\s]+\|[\s]*(\|.+\|[\s]*)+)"
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    result = []
    for t in tables:
        table_text = t[0].strip()
        if len(table_text) > max_chars:
            table_text = table_text[:max_chars] + "\n... [truncated]"
        result.append(table_text)
    return result


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


def full_pipeline_v2(query: str, timeout_seconds: int = 180):
    """
    RAG пайплайн с мультимодальным поиском и реранкингом.

    Возвращает: (answer, answer_normalized, total_time)
      answer            — ответ после parse_answer (для отчёта)
      answer_normalized — ответ после normalize_answer (для метрик F1/EM)
      total_time        — время выполнения

    Шаги:
      1-2.  Тип вопроса + домен документа
      3.    Поиск по номеру Figure/Table (приоритетный путь)
      4.    Кодирование запроса эмбеддером
      5.    Гибридный поиск страниц (BM25 + vector + RRF)
      6.    Поиск text chunks + изображений из отдельных индексов
      7.    Реранкинг всех модальностей через Nemotron
      8.    Лимиты контекста по типу × домену
      9.    Расширение соседними страницами
      10.   Сборка контекста (PNG страницы + Markdown таблицы)
      11.   Генерация ответа VLM с таймаутом
      12.   Постпроцессинг: parse + normalize
    """
    start_time = time.time()

    print(f"\n[1/12] Question: {query[:80]}...")

    # ── Шаг 1-2: Тип и домен ──────────────────────────────────────────
    q_type = determine_question_type(query)
    print(f"[2/12] Type: {q_type}")

    # ── Шаг 3: Поиск по номеру Figure/Table ───────────────────────────
    figure_pages = search_by_figure_number(query, bm25)
    print(f"[3/12] Found {len(figure_pages)} pages by figure/table number")

    # ── Шаг 4: Кодируем запрос ────────────────────────────────────────
    query_emb = encode_query(query)

    # ── Шаг 5: Гибридный поиск страниц ───────────────────────────────
    page_results = search_pages(query, query_emb, top_k=30)

    # Figure-страницы вставляем в начало с максимальным приоритетом
    if figure_pages:
        existing_keys = {f"{r['folder']}_{r['page']}" for r in page_results}
        for fp in figure_pages:
            key = f"{fp['folder']}_{fp['page']}"
            if key not in existing_keys:
                page_results.insert(0, fp)

    # Шаг 5.1: ОПРЕДЕЛЯЕМ ДОМЕН ПО ТОП-1 СТРАНИЦ

    domain = "academic"
    if page_results:
        folder_name = page_results[0].get("folder", "")
        domain = domain_from_folder(folder_name)
    print(f"[5/12] Domain from folder: {domain}")

    # ── Шаг 6: Поиск текста и изображений ────────────────────────────
    text_results = search_text(query_emb, top_k=50)
    image_results = search_images(query_emb, top_k=50)
    print(
        f"[4/12] Found: {len(page_results)} pages, "
        f"{len(text_results)} texts, {len(image_results)} images"
    )

    # ── Шаг 7: Реранкинг ─────────────────────────────────────────────
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
        f"[5/12] Reranked: {len(page_reranked)} pages, "
        f"{len(text_reranked)} texts, {len(image_reranked)} images"
    )

    # ── Шаг 8: Лимиты контекста ──────────────────────────────────────
    limits = get_dynamic_limits(q_type, domain)
    print(
        f"[6/12] Limits: pages={limits['max_pages']}, "
        f"tables={limits['max_tables']}, "
        f"images={limits['max_images']} "
        f"(type={q_type}, domain={domain})"
    )

    # ── Шаг 9: Расширяем соседними страницами ─────────────────────────
    # Берём топ-N по реранкингу, добавляем ±1 соседа для каждой
    # Финальный лимит: MAX_PAGES_HARD_LIMIT (защита от переполнения VLM)
    top_pages = page_reranked[: limits["max_pages"]]
    expanded_pages = expand_with_neighbors(top_pages, max_neighbors_per_page=1)
    # Жёсткий потолок: не более MAX_PAGES_HARD_LIMIT страниц в VLM
    expanded_pages = expanded_pages[:MAX_PAGES_HARD_LIMIT]

    # ── Шаг 10: Сборка контекста ──────────────────────────────────────
    # Определяем, числовой ли вопрос
    is_numeric = any(
        kw in query.lower()
        for kw in [
            "how many",
            "how much",
            "what is the total",
            "calculate",
            "sum",
            "accuracy",
            "score",
            "f1",
            "bleu",
            "perplexity",
            "percentage",
            "performance score",
            "value",
            "number",
        ]
    )

    context_images = []
    context_texts = []

    if q_type == "table" and is_numeric:
        # ПРИОРИТЕТ: Markdown таблицы идут ПЕРВЫМИ
        # 1. Markdown таблицы (основной источник)
        if limits["max_tables"] > 0:
            for cand in text_reranked[: limits["max_tables"] * 2]:
                markdown_text = cand.get("text", "")
                if not markdown_text:
                    continue
                tables = extract_tables_from_markdown(markdown_text, max_chars=800)
                for table in tables[: limits["max_tables"]]:
                    context_texts.append(table)

        # 2. PNG страницы (только 1 страница, для контекста)
        for cand in expanded_pages[:1]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except Exception:
                continue

        print(
            f"[7/12] Numeric table: {len(context_texts)} markdown tables (primary), {len(context_images)} PNG pages (secondary)"
        )

    else:
        # Обычный путь (для не-числовых или не-табличных вопросов)
        # PNG страницы
        for cand in expanded_pages:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except Exception:
                continue

        # Markdown таблицы из text index (для table-вопросов)
        if limits["max_tables"] > 0 and q_type in ("table", "text"):
            for cand in text_reranked[: limits["max_tables"] * 2]:
                markdown_text = cand.get("text", "")
                if not markdown_text:
                    continue
                tables = extract_tables_from_markdown(markdown_text, max_chars=800)
                for table in tables[: limits["max_tables"]]:
                    context_texts.append(table)

        # Отдельные изображения из image index (для image/formula вопросов)
        if limits["max_images"] > 0 and q_type in ("image", "formula"):
            for cand in image_reranked[: limits["max_images"]]:
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    context_images.append(img)
                except Exception:
                    continue

    print(f"[7/12] Context: {len(context_images)} images, {len(context_texts)} text chunks")

    if not context_images and not context_texts:
        return "NOT FOUND", "not found", time.time() - start_time

    # ── Шаг 11: Генерация ─────────────────────────────────────────────
    print(f"[8/12] Generating (timeout={timeout_seconds}s)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if context_images and context_texts:
            combined_text = "\n\n".join(context_texts)
            raw_answer = qwen_generator.generate_answer(query, context_images, combined_text)
        elif context_images:
            raw_answer = qwen_generator.generate_answer(query, context_images)
        else:
            combined_text = "\n\n".join(context_texts)
            raw_answer = qwen_generator.generate_answer(query, None, combined_text)
        print("[9/12] Generation completed")
    except TimeoutError:
        raw_answer = "TIMEOUT"
        print(f"[9/12] TIMEOUT after {timeout_seconds}s")
    finally:
        signal.alarm(0)

    # ── Шаг 12: Постпроцессинг ────────────────────────────────────────
    answer = parse_answer(raw_answer)
    answer_normalized = normalize_answer(answer)

    total_time = time.time() - start_time
    print(f"[10/12] Total time: {total_time:.2f}s")
    print(f"[11/12] Raw:        {raw_answer[:120]}")
    print(f"[12/12] Processed:  {answer[:120]}")

    return answer, answer_normalized, total_time
