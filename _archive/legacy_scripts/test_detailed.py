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

data_path = project_root / "data" / "datasets" / "docbench"
index_dir = project_root / "index"
cache_dir = project_root / "cache" / "modules"
docling_dir = project_root / "data" / "docling_markdown_fast"


class DetailedTimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise DetailedTimeoutError("Generation took too long")


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


print("=" * 80)
print("LOADING MODELS")
print("=" * 80)

print("1. Loading Qwen3-VL-Embedding on CPU...")
embed_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_embed,
).eval()
embed_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True
)

print("2. Loading page index...")
page_index = faiss.read_index(str(index_dir / "pages_qwen3.index"))
with open(index_dir / "metadata_qwen3.json", "r", encoding="utf-8") as f:
    page_metadata = json.load(f)
print(f"   Page index: {page_index.ntotal} vectors")

print("3. Loading text index...")
text_index = faiss.read_index(str(project_root / "index_text_full" / "index.faiss"))
with open(project_root / "index_text_full" / "metadata.pkl", "rb") as f:
    text_metadata = pickle.load(f)
print(f"   Text index: {text_index.ntotal} vectors")

print("4. Loading image index...")
image_index = faiss.read_index(str(project_root / "index_images_full" / "index.faiss"))
with open(project_root / "index_images_full" / "metadata.pkl", "rb") as f:
    image_metadata = pickle.load(f)
print(f"   Image index: {image_index.ntotal} vectors")

print("5. Loading Nemotron Rerank on GPU...")
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

print("6. Loading BM25...")
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
print("   BM25 loaded")

print("7. Loading Qwen3-VL generator...")
qwen_generator = create_table_generator(device=device_llm)

print("\n" + "=" * 80)
print("MODELS LOADED")
print("=" * 80 + "\n")


def encode_query(query):
    with torch.no_grad():
        inputs = embed_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device_embed) for k, v in inputs.items()}
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def search_pages(query, emb, top_k=30):
    start = time.time()
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
    elapsed = time.time() - start
    return candidates, elapsed


def search_text(emb, top_k=50):
    start = time.time()
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
    elapsed = time.time() - start
    return results, elapsed


def search_images(emb, top_k=50):
    start = time.time()
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
    elapsed = time.time() - start
    return results, elapsed


def expand_with_neighbors(candidates, max_pages=15):
    start = time.time()
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
    elapsed = time.time() - start
    return list(unique.values())[:max_pages], elapsed


def rerank_candidates(query, candidates, model, processor, device, name=""):
    if not candidates:
        return [], 0
    start = time.time()
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

    result = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
    elapsed = time.time() - start
    return result, elapsed


def full_pipeline_detailed(query, timeout_seconds=200):
    print("\n" + "=" * 80)
    print(f"QUESTION: {query}")
    print("=" * 80)

    total_start = time.time()
    timing = {}

    # Тип вопроса
    q_type = determine_question_type(query)
    print(f"\n[STEP 1] Question type detection: {q_type}")

    # Кодирование запроса
    t0 = time.time()
    query_emb = encode_query(query)
    timing["encode_query"] = time.time() - t0
    print(f"  -> Encode query: {timing['encode_query']:.2f}s")

    # Поиск страниц
    t0 = time.time()
    page_results, t_search_pages = search_pages(query, query_emb, top_k=30)
    timing["search_pages"] = t_search_pages
    print(f"\n[STEP 2] Search pages (BM25 + Vector + RRF): {timing['search_pages']:.2f}s")
    print(f"  -> Found {len(page_results)} pages")

    # Поиск текста
    t0 = time.time()
    text_results, t_search_text = search_text(query_emb, top_k=50)
    timing["search_text"] = t_search_text
    print(f"\n[STEP 3] Search text index: {timing['search_text']:.2f}s")
    print(f"  -> Found {len(text_results)} text blocks")

    # Поиск изображений
    t0 = time.time()
    image_results, t_search_images = search_images(query_emb, top_k=50)
    timing["search_images"] = t_search_images
    print(f"\n[STEP 4] Search image index: {timing['search_images']:.2f}s")
    print(f"  -> Found {len(image_results)} images")

    # Реранкинг страниц
    t0 = time.time()
    page_reranked, t_rerank_pages = rerank_candidates(
        query, page_results, rerank_model, rerank_processor, device_rerank, "pages"
    )
    timing["rerank_pages"] = t_rerank_pages
    print(f"\n[STEP 5] Rerank pages (Nemotron): {timing['rerank_pages']:.2f}s")
    print(f"  -> Reranked {len(page_reranked)} pages")
    for i, p in enumerate(page_reranked[:5]):
        print(
            f"     Rank {i+1}: folder={p['folder']}, page={p.get('page', 'N/A')}, score={p.get('rerank_score', 0):.4f}"
        )

    # Реранкинг текста
    t0 = time.time()
    text_reranked, t_rerank_text = rerank_candidates(
        query, text_results, rerank_model, rerank_processor, device_rerank, "text"
    )
    timing["rerank_text"] = t_rerank_text
    print(f"\n[STEP 6] Rerank text blocks: {timing['rerank_text']:.2f}s")
    print(f"  -> Reranked {len(text_reranked)} text blocks")

    # Реранкинг изображений
    t0 = time.time()
    image_reranked, t_rerank_images = rerank_candidates(
        query, image_results, rerank_model, rerank_processor, device_rerank, "images"
    )
    timing["rerank_images"] = t_rerank_images
    print(f"\n[STEP 7] Rerank images: {timing['rerank_images']:.2f}s")
    print(f"  -> Reranked {len(image_reranked)} images")

    # Расширение соседними страницами
    top_pages = page_reranked[:3]
    t0 = time.time()
    expanded_pages, t_expand = expand_with_neighbors(top_pages, max_pages=15)
    timing["expand_pages"] = t_expand
    print(f"\n[STEP 8] Expand with neighbors: {timing['expand_pages']:.2f}s")
    print(f"  -> Top 3 pages + neighbors = {len(expanded_pages)} pages")

    # Загрузка изображений страниц
    t0 = time.time()
    context_images = []
    for cand in expanded_pages[:15]:
        try:
            img = Image.open(cand["path"]).convert("RGB")
            context_images.append(img)
        except Exception as e:
            print(f"     Warning: could not load {cand.get('path', 'unknown')}: {e}")
    timing["load_images"] = time.time() - t0
    print(f"\n[STEP 9] Load page images: {timing['load_images']:.2f}s")
    print(f"  -> Loaded {len(context_images)} images")
    print(f"  -> Each image size: {context_images[0].size if context_images else 'N/A'}")

    # Сбор текстового контекста (таблиц)
    context_texts = []
    text_content_preview = ""
    if q_type == "table":
        t0 = time.time()
        for cand in text_reranked[:2]:
            markdown_text = cand.get("text", "")
            if markdown_text:
                tables = extract_tables_from_markdown(markdown_text, max_chars=500)
                for table in tables[:1]:
                    context_texts.append(table)
        timing["extract_tables"] = time.time() - t0
        print(f"\n[STEP 10] Extract tables from text blocks: {timing['extract_tables']:.2f}s")
        print(f"  -> Extracted {len(context_texts)} tables")
        if context_texts:
            text_content_preview = context_texts[0][:300]
            print(f"  -> Table preview (first 300 chars):\n{text_content_preview}...")
    elif q_type == "image":
        t0 = time.time()
        for cand in image_reranked[:3]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        timing["add_images"] = time.time() - t0
        print(f"\n[STEP 10] Add images for image question: {timing['add_images']:.2f}s")
        print(f"  -> Added {len(image_reranked[:3])} images")
    elif q_type == "formula":
        t0 = time.time()
        for cand in image_reranked[:3]:
            try:
                img = Image.open(cand["path"]).convert("RGB")
                context_images.append(img)
            except:
                continue
        timing["add_formulas"] = time.time() - t0
        print(f"\n[STEP 10] Add formula images: {timing['add_formulas']:.2f}s")
        print(f"  -> Added {len(image_reranked[:3])} formula images")

    # Итоговый контекст
    print("\n" + "=" * 80)
    print("FINAL CONTEXT SENT TO VLM")
    print("=" * 80)
    print(f"  Images (PNG): {len(context_images)}")
    print(f"  Text chunks (Markdown tables): {len(context_texts)}")
    if text_content_preview:
        print(f"\n  Text content preview:\n{text_content_preview}")
    print("\n  Images list:")
    for i, img in enumerate(context_images[:5]):
        print(f"    Image {i+1}: size {img.size}")

    # Генерация ответа
    print("\n" + "=" * 80)
    print("GENERATING ANSWER")
    print("=" * 80)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    t0 = time.time()
    try:
        if context_images and context_texts:
            combined_text = "\n\n".join(context_texts)
            answer = qwen_generator.generate_answer(query, context_images, combined_text)
        elif context_images:
            answer = qwen_generator.generate_answer(query, context_images)
        else:
            combined_text = "\n\n".join(context_texts)
            answer = qwen_generator.generate_answer(query, None, combined_text)
        timing["generation"] = time.time() - t0
        print(f"\n  Generation time: {timing['generation']:.2f}s")
        print(f"  Generated answer: {answer}")
    except DetailedTimeoutError:
        timing["generation"] = time.time() - t0
        answer = "TIMEOUT"
        print(f"\n  Generation time: {timing['generation']:.2f}s (TIMEOUT)")
        print(f"  Generated answer: TIMEOUT")
    finally:
        signal.alarm(0)

    total_time = time.time() - total_start
    timing["total"] = total_time

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    for step, t in timing.items():
        print(f"  {step:20s}: {t:6.2f}s")

    return (
        answer,
        timing,
        {
            "question_type": q_type,
            "num_images": len(context_images),
            "num_texts": len(context_texts),
            "generated_answer": answer,
        },
    )


# ============================================
# ТЕСТОВЫЕ ВОПРОСЫ (10 штук с известными ответами)
# ============================================

test_questions = [
    ("What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?", "65%"),
    ("How many documents are there in the training set of the Linked WikiText-2 Corpus?", "600"),
    ("Which language model has the lowest Perplexity (PPL) according to Table 3?", "KGLM"),
    (
        "Which dataset experienced the largest decrease in BLEU score after alignment according to Table 4?",
        "GL→EN",
    ),
    (
        "What is the total number of sentences in the training sets for Romance languages as given in Table 1?",
        "61802",
    ),
    (
        "What is the performance score for Entity Recognition when multitasked with Coreference Resolution?",
        "67.5",
    ),
    ("Which model has the highest F1 score for entity recognition on the Test set?", "SciIE"),
    ("Which model achieved the highest F1 score in span identification?", "SciIE"),
    (
        "What is the test set accuracy of BERT (Large) as reported in the best run according to Table 1?",
        "77%",
    ),
    (
        "What function is used to determine a probability distribution over the two warrants in the proposed architecture?",
        "softmax",
    ),
]

print("\n" + "=" * 80)
print("STARTING DETAILED TEST ON 10 QUESTIONS")
print("=" * 80)

all_results = []

for i, (question, expected) in enumerate(test_questions, 1):
    print(f"\n\n{'#' * 80}")
    print(f"QUESTION {i}/10")
    print(f"{'#' * 80}")

    answer, timing, info = full_pipeline_detailed(question, timeout_seconds=200)

    all_results.append(
        {
            "question": question,
            "expected": expected,
            "generated": answer,
            "timing": timing,
            "info": info,
        }
    )

    print(f"\n{'=' * 60}")
    print(f"RESULT {i}/10")
    print(f"  Expected: {expected}")
    print(f"  Generated: {answer}")
    print(f"  Total time: {timing['total']:.2f}s")
    print(f"{'=' * 60}")

print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

total_times = []
for r in all_results:
    total_times.append(r["timing"]["total"])
    print(f"\nQ: {r['question'][:80]}...")
    print(f"  Expected: {r['expected']}")
    print(f"  Generated: {r['generated']}")
    print(f"  Time: {r['timing']['total']:.2f}s")
    print(f"  Images: {r['info']['num_images']}, Texts: {r['info']['num_texts']}")

avg_time = sum(total_times) / len(total_times)
print(f"\n{'=' * 80}")
print(f"Average time per question: {avg_time:.2f}s")
print(f"Total time for 10 questions: {sum(total_times):.2f}s")
print(f"{'=' * 80}")
