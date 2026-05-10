"""
RAG Evaluation Pipeline v2 — ДИАГНОСТИЧЕСКАЯ ВЕРСИЯ
Отличается от full_pipeline_v2 детальным выводом контекста, передаваемого в VLM.
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

FOLDER_DOMAIN_CACHE = {}


def load_domains_from_reports():
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
    return FOLDER_DOMAIN_CACHE.get(folder_name, "academic")


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


def normalize_answer(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_answer(raw_output: str) -> str:
    if not raw_output:
        return "NOT FOUND"
    text = re.sub(r"^assistant\s*\n", "", raw_output, flags=re.IGNORECASE).strip()
    lower = text.lower()
    if "internal thought:" in lower:
        idx = lower.find("internal thought:")
        after_thought = text[idx:]
        lines = [l.strip() for l in after_thought.split("\n") if l.strip()]
        content_lines = [l for l in lines[1:] if not l.lower().startswith("internal thought")]
        if content_lines:
            return content_lines[-1]
    return text.strip() if text.strip() else "NOT FOUND"


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

print("Loading document domains...")
load_domains_from_reports()


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
    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]


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
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


# ============================================================
# ДИАГНОСТИЧЕСКАЯ ФУНКЦИЯ С ВЫВОДОМ КОНТЕКСТА
# ============================================================


def full_pipeline_diagnose(query, timeout_seconds=180):
    start_time = time.time()

    print(f"\n[Q] {query[:100]}...")

    q_type = determine_question_type(query)
    print(f"[1] Type: {q_type}")

    figure_pages = []  # search_by_figure_number(query, bm25) - упрощённо
    print(f"[2] Figure/table pages found: {len(figure_pages)}")

    query_emb = encode_query(query)

    page_results = search_pages(query, query_emb, top_k=30)

    if figure_pages:
        existing_keys = {f"{r['folder']}_{r['page']}" for r in page_results}
        for fp in figure_pages:
            key = f"{fp['folder']}_{fp['page']}"
            if key not in existing_keys:
                page_results.insert(0, fp)

    domain = "academic"
    if page_results:
        folder_name = str(page_results[0].get("folder", ""))
        domain = domain_from_folder(folder_name)
    print(f"[5] Domain: {domain}")

    text_results = search_text(query_emb, top_k=50)
    image_results = search_images(query_emb, top_k=50)
    print(
        f"[6] Candidates: {len(page_results)} pages, {len(text_results)} texts, {len(image_results)} images"
    )

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
        f"[7] Reranked: {len(page_reranked)} pages, {len(text_reranked)} texts, {len(image_reranked)} images"
    )

    # Лимиты контекста (фиксированные для диагностики)
    max_pages = 2
    max_tables = 2
    max_images = 0

    top_pages = page_reranked[:max_pages]
    expanded_pages = expand_with_neighbors(top_pages, max_pages=15)
    expanded_pages = expanded_pages[:6]

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
        if max_tables > 0:
            for cand in text_reranked[: max_tables * 2]:
                markdown_text = cand.get("text", "")
                if markdown_text:
                    tables = extract_tables_from_markdown(markdown_text, max_chars=800)
                    for table in tables[:max_tables]:
                        context_texts.append(table)
        for cand in expanded_pages[:1]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        print(f"[10] Numeric table: {len(context_texts)} md tables + {len(context_images)} PNG")
    else:
        for cand in expanded_pages:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        if max_tables > 0 and q_type in ("table", "text"):
            for cand in text_reranked[: max_tables * 2]:
                markdown_text = cand.get("text", "")
                if not markdown_text:
                    continue
                tables = extract_tables_from_markdown(markdown_text, max_chars=800)
                for table in tables[:max_tables]:
                    context_texts.append(table)
        if max_images > 0 and q_type in ("image", "formula"):
            for cand in image_reranked[:max_images]:
                try:
                    img = Image.open(cand["path"]).convert("RGB")
                    context_images.append(img)
                except:
                    continue

    # ============================================================
    # ДИАГНОСТИЧЕСКИЙ ВЫВОД КОНТЕКСТА
    # ============================================================
    print("\n" + "=" * 100)
    print("📤 CONTEXT SENT TO VLM")
    print("=" * 100)

    print(f"\n📌 Pages sent ({len(context_images)} images):")
    for i, cand in enumerate(expanded_pages[:max_pages]):
        print(
            f"   [{i+1}] Folder: {cand.get('folder', '?')}, Page: {cand.get('page', '?')}, Path: {cand.get('path', '?')}"
        )

    print(f"\n📌 Markdown tables sent ({len(context_texts)} chunks):")
    for i, text in enumerate(context_texts):
        print(f"\n   [TABLE {i+1}]")
        print(f"   {'-' * 96}")
        print(text)
        print(f"   {'-' * 96}")

    print("\n" + "=" * 100)
    print("✅ END OF CONTEXT")
    print("=" * 100)

    print(f"\n[11] Generating (timeout={timeout_seconds}s)...")
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
        print("[11] Generation completed")
    except TimeoutError:
        raw_answer = "TIMEOUT"
        print(f"[11] TIMEOUT after {timeout_seconds}s")
    finally:
        signal.alarm(0)

    answer = parse_answer(raw_answer)
    answer_normalized = normalize_answer(answer)

    total_time = time.time() - start_time
    print(f"[12] Time: {total_time:.2f}s")
    print(f"[12] Raw:       {raw_answer[:120]}")
    print(f"[12] Processed: {answer[:120]}")

    return answer, answer_normalized, total_time


if __name__ == "__main__":
    test_queries = [
        "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
    ]
    for q in test_queries:
        answer, answer_norm, total_time = full_pipeline_diagnose(q)
        print(f"\nFINAL ANSWER: {answer}")
