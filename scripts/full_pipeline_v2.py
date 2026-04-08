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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.modules.bm25_module import BM25Module
from src.core.generators.qwen_vl_generator import create_table_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_llm = "cuda"
device_rerank = "cuda"
device_embed = "cpu"

print(f"Using device for LLM: {device_llm}")
print(f"Using device for Rerank: {device_rerank}")
print(f"Using device for Embedding: {device_embed}")

data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"
cache_dir = project_root / "cache" / "modules"
docling_dir = project_root / "data" / "docling_markdown_fast"


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Generation took too long")


# ============================================
# КЛЮЧЕВЫЕ СЛОВА
# ============================================

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
    "calculate",
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


def determine_question_type(query):
    q = query.lower()
    if any(kw in q for kw in TABLE_KEYWORDS):
        return "table"
    if any(kw in q for kw in IMAGE_KEYWORDS):
        return "image"
    if any(kw in q for kw in FORMULA_KEYWORDS):
        return "formula"
    return "text"


def extract_tables_from_markdown(markdown_text, max_chars=500):
    table_pattern = r"(\|.+\|[\s]*[\|\-\s]+\|[\s]*(\|.+\|[\s]*)+)"
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
    result = []
    for t in tables:
        table_text = t[0]
        if len(table_text) > max_chars:
            table_text = table_text[:max_chars]
        result.append(table_text)
    return result


# ============================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================

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

print("Loading page index (Qwen3-VL)...")
page_index = faiss.read_index(str(index_dir / "pages_qwen3.index"))
with open(index_dir / "metadata_qwen3.json", "r", encoding="utf-8") as f:
    page_metadata = json.load(f)
print(f"Page index: {page_index.ntotal} vectors")

print("Loading text index (Docling)...")
text_index = faiss.read_index(str(project_root / "index_text_full" / "index.faiss"))
with open(project_root / "index_text_full" / "metadata.pkl", "rb") as f:
    text_metadata = pickle.load(f)
print(f"Text index: {text_index.ntotal} vectors")

print("Loading image index (PyMuPDF4LLM)...")
image_index = faiss.read_index(str(project_root / "index_images_full" / "index.faiss"))
with open(project_root / "index_images_full" / "metadata.pkl", "rb") as f:
    image_metadata = pickle.load(f)
print(f"Image index: {image_index.ntotal} vectors")

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

print("Loading BM25...")
bm25 = BM25Module(name="bm25", language="multilingual", k1=2.5, b=0.9)
bm25_path = cache_dir / "bm25"
if bm25_path.exists():
    bm25.load(str(cache_dir))
else:
    print("Building BM25...")
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

print("Loading Qwen3-VL generator...")
qwen_generator = create_table_generator(device=device_llm)


# ============================================
# ФУНКЦИИ ПОИСКА
# ============================================


def encode_query(query):
    with torch.no_grad():
        inputs = embed_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device_embed) for k, v in inputs.items()}
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def search_pages(query, emb, top_k=30):
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

    candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return candidates


def search_text(emb, top_k=50):
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


def search_images(emb, top_k=50):
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


def expand_with_neighbors(candidates, max_pages=15):
    expanded = []
    for cand in candidates:
        expanded.append(cand)
        if cand["type"] != "page":
            continue
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
        key = f"{cand['folder']}_{cand.get('page', 0)}_{cand['type']}"
        if key not in unique:
            unique[key] = cand
    return list(unique.values())[:max_pages]


def rerank_candidates(query, candidates, model, processor, device):
    if not candidates:
        return []
    reranked = []
    batch_size = 4
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        examples = []
        for cand in batch:
            if cand["type"] == "page":
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    examples.append({"question": query, "doc_text": "", "doc_image": img})
                except:
                    continue
            elif cand["type"] == "image":
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    examples.append({"question": query, "doc_text": "", "doc_image": img})
                except:
                    continue
            elif cand["type"] == "text":
                text_content = cand.get("text", "")
                if not text_content:
                    continue
                examples.append(
                    {"question": query, "doc_text": text_content[:2000], "doc_image": None}
                )

        if not examples:
            continue

        batch_dict = processor.process_queries_documents_crossencoder(examples)
        batch_dict = {k: v.to(device) for k, v in batch_dict.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            logits = model(**batch_dict, return_dict=True).logits
        for j, logit in enumerate(logits):
            if j < len(batch):
                batch[j]["rerank_score"] = float(torch.sigmoid(logit).item())
                reranked.append(batch[j])

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================


def full_pipeline_v2(query, timeout_seconds=90):
    start_time = time.time()

    print(f"\nQuery: {query}")

    q_type = determine_question_type(query)
    print(f"Question type: {q_type}")

    query_emb = encode_query(query)

    print("Searching pages...")
    page_results = search_pages(query, query_emb, top_k=30)

    print("Searching text index...")
    text_results = search_text(query_emb, top_k=50)

    print("Searching image index...")
    image_results = search_images(query_emb, top_k=50)

    print(
        f"\nFound: {len(page_results)} pages, {len(text_results)} texts, {len(image_results)} images"
    )

    print("\nReranking pages...")
    page_reranked = rerank_candidates(
        query, page_results, rerank_model, rerank_processor, device_rerank
    )

    print("Reranking texts...")
    text_reranked = rerank_candidates(
        query, text_results, rerank_model, rerank_processor, device_rerank
    )

    print("Reranking images...")
    image_reranked = rerank_candidates(
        query, image_results, rerank_model, rerank_processor, device_rerank
    )

    top_pages = page_reranked[:3]
    expanded_pages = expand_with_neighbors(top_pages, max_pages=15)

    context_images = []
    context_texts = []

    # Добавляем страницы PNG (всегда)
    for cand in expanded_pages[:15]:
        try:
            img = Image.open(cand["path"]).convert("RGB")
            context_images.append(img)
        except:
            continue

    if q_type == "table":
        # Добавляем таблицы из текстовых блоков (до 2 таблиц, 500 символов)
        for cand in text_reranked[:2]:
            markdown_text = cand.get("text", "")
            if markdown_text:
                tables = extract_tables_from_markdown(markdown_text, max_chars=500)
                for table in tables[:1]:
                    context_texts.append(table)
        print(f"Added {len(context_texts)} tables")

    elif q_type == "image":
        for cand in image_reranked[:3]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        print(f"Added {len(image_reranked[:3])} images")

    elif q_type == "formula":
        for cand in image_reranked[:3]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        print(f"Added {len(image_reranked[:3])} formula images")

    print(f"\n=== SENT TO VLM ===")
    print(f"Images: {len(context_images)}")
    print(f"Texts: {len(context_texts)}")

    if not context_images and not context_texts:
        return "NOT FOUND", time.time() - start_time

    print(f"\nGenerating answer (timeout: {timeout_seconds}s)...")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if context_images and context_texts:
            answer = qwen_generator.generate_answer(query, context_images, context_texts)
        elif context_images:
            answer = qwen_generator.generate_answer(query, context_images)
        else:
            answer = qwen_generator.generate_answer(query, None, context_texts)
    except TimeoutError:
        answer = "TIMEOUT - Generation took too long"
        print(f"Timeout after {timeout_seconds}s")
    finally:
        signal.alarm(0)

    total_time = time.time() - start_time
    print(f"Time: {total_time:.2f}s")
    print(f"Answer: {answer}")

    return answer, total_time


# ============================================
# ТЕСТ
# ============================================

if __name__ == "__main__":
    test_queries = [
        "What is the aligned BLEU score for GL → EN?",
        "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
        "Which model achieved the highest accuracy on the random split according to Table 5?",
    ]

    for q in test_queries:
        answer, _ = full_pipeline_v2(q)
        print(f"\n{'=' * 60}")
        print(f"FINAL ANSWER: {answer}")
        print(f"{'=' * 60}\n")
