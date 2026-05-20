from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "has",
    "have",
    "had",
    "which",
    "what",
    "when",
    "where",
    "who",
    "how",
    "according",
    "table",
    "figure",
    "score",
    "accuracy",
    "using",
    "resulting",
    "answer",
}


@dataclass(frozen=True)
class PageTextMatch:
    page: int
    exact_answer: bool
    number_recall: float
    keyword_recall: float
    matched_numbers: list[str]
    matched_keywords: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ColPali/ColQwen retrieval against answer-derived oracle pages."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3_merged"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_merged_clean")
    parser.add_argument("--model-id", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_colvision_oracle_pages_clean.json")
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numbers(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"\d+(?:\.\d+)?%?", normalized)


def extract_keywords(text: str) -> list[str]:
    normalized = normalize_text(text)
    words = re.findall(r"[a-z][a-z0-9_+\-]*", normalized)
    return [word for word in words if len(word) >= 3 and word not in STOP_WORDS]


def load_pages_text(folder: Path) -> list[dict]:
    path = folder / "extracted" / "pages_text.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_answer_pages(folder: Path, answer: str, evidence: str = "") -> list[PageTextMatch]:
    pages = load_pages_text(folder)
    answer_norm = normalize_text(answer)
    numbers = extract_numbers(answer)
    keywords = extract_keywords(answer)
    evidence_keywords = extract_keywords(evidence)
    if evidence_keywords:
        keywords = list(dict.fromkeys([*keywords, *evidence_keywords]))

    matches = []
    for row in pages:
        page_text = normalize_text(row.get("text", ""))
        page_numbers = set(extract_numbers(page_text))
        page_words = set(extract_keywords(page_text))
        matched_numbers = [
            num for num in numbers if num in page_numbers or num.rstrip("%") in page_numbers
        ]
        matched_keywords = [word for word in keywords if word in page_words]
        exact = bool(answer_norm and answer_norm in page_text)
        number_recall = len(set(matched_numbers)) / len(set(numbers)) if numbers else 0.0
        keyword_recall = len(set(matched_keywords)) / len(set(keywords)) if keywords else 0.0
        if exact or number_recall > 0 or keyword_recall >= 0.25:
            matches.append(
                PageTextMatch(
                    page=int(row.get("page", 0)),
                    exact_answer=exact,
                    number_recall=number_recall,
                    keyword_recall=keyword_recall,
                    matched_numbers=matched_numbers,
                    matched_keywords=matched_keywords[:20],
                )
            )
    matches.sort(
        key=lambda item: (item.exact_answer, item.number_recall, item.keyword_recall), reverse=True
    )
    return matches


def load_questions(data_dir: Path, types: set[str]) -> list[dict[str, str]]:
    questions = []
    for jsonl_file in sorted(data_dir.glob("*/*_qa.jsonl")):
        folder = jsonl_file.parent.name
        if not folder.isdigit():
            continue
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if types and row.get("type") not in types:
                    continue
                questions.append(
                    {
                        "folder": folder,
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "evidence": row.get("evidence", ""),
                        "type": row.get("type", ""),
                    }
                )
    return questions


def load_model_and_processor(model_id: str, device: str) -> tuple[Any, Any, str]:
    import torch

    lower = model_id.lower()
    if "colqwen2.5" in lower or "colqwen2_5" in lower:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        model_cls = ColQwen2_5
        processor_cls = ColQwen2_5_Processor
        family = "colqwen2.5"
    elif "colqwen2" in lower:
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        model_cls = ColQwen2
        processor_cls = ColQwen2Processor
        family = "colqwen2"
    elif "colpali" in lower:
        from colpali_engine.models import ColPali, ColPaliProcessor

        model_cls = ColPali
        processor_cls = ColPaliProcessor
        family = "colpali"
    else:
        raise ValueError("Unsupported model family.")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
    processor = processor_cls.from_pretrained(model_id)
    return model, processor, family


def split_embeddings(output: Any) -> list[Any]:
    if isinstance(output, list):
        return output
    if hasattr(output, "detach"):
        if output.ndim == 3:
            return [output[i] for i in range(output.shape[0])]
        if output.ndim == 2:
            return [output]
    raise TypeError(f"Unsupported embedding output type/shape: {type(output)}")


def encode_query(model: Any, processor: Any, query: str, device: str) -> Any:
    import torch

    batch = processor.process_queries([query]).to(device)
    with torch.no_grad():
        output = model(**batch)
    return split_embeddings(output)[0]


def score_docs(processor: Any, query_embedding: Any, docs: list[Any], device: str) -> list[float]:
    import torch

    dtype = docs[0].dtype if docs else query_embedding.dtype
    query = query_embedding.to(device=device, dtype=dtype)
    doc_batch = [doc.to(device=device, dtype=dtype) for doc in docs]
    with torch.no_grad():
        scores = processor.score_multi_vector([query], doc_batch)
    if hasattr(scores, "detach"):
        values = scores.detach().float().cpu().numpy()
    else:
        values = np.asarray(scores, dtype=np.float32)
    return [float(value) for value in values.reshape(-1)]


def recall_dict(hits: dict[int, int], total: int) -> dict[str, float]:
    denominator = max(total, 1)
    return {f"R@{k}": hits[k] / denominator for k in sorted(hits)}


def main() -> None:
    args = parse_args()
    import torch

    metadata_path = args.index_dir / f"metadata_{args.index_name}.json"
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    records_by_shard: dict[str, list[dict]] = defaultdict(list)
    for record in metadata:
        records_by_shard[record["shard"]].append(record)

    model, processor, family = load_model_and_processor(args.model_id, args.device)
    questions = load_questions(args.data_dir, set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    cutoffs = sorted({1, 5, 10, 30, args.top_k})
    cutoffs = [k for k in cutoffs if k <= args.top_k]
    folder_hits = {k: 0 for k in cutoffs}
    oracle_hits = {k: 0 for k in cutoffs}
    oracle_available = 0
    folder_found_oracle_missed = 0
    folder_and_oracle_found = 0
    no_oracle_rows = []
    failure_examples = []
    success_examples = []
    rows = []

    for qi, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, processor, question["question"], args.device)
        heap: list[tuple[float, int, dict]] = []

        for shard_name, shard_records in records_by_shard.items():
            shard_embeddings = torch.load(
                args.index_dir / "shards" / shard_name, map_location="cpu"
            )
            for offset in range(0, len(shard_records), args.score_batch_size):
                batch_records = shard_records[offset : offset + args.score_batch_size]
                docs = [shard_embeddings[int(record["shard_offset"])] for record in batch_records]
                scores = score_docs(processor, query_embedding, docs, args.device)
                for score, record in zip(scores, batch_records):
                    item = (float(score), int(record["index"]), record)
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            del shard_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(heap, key=lambda item: item[0], reverse=True)
        candidates = []
        ranked_folders = []
        ranked_page_keys = []
        for rank, (score, _, record) in enumerate(ranked, start=1):
            folder = str(record["folder"])
            page = int(record["page"])
            ranked_folders.append(folder)
            ranked_page_keys.append(f"{folder}_{page}")
            candidates.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(record["index"]),
                    "folder": folder,
                    "page": page,
                    "path": record["path"],
                }
            )

        expected = str(question["folder"])
        folder_rank = ranked_folders.index(expected) + 1 if expected in ranked_folders else None
        for k in folder_hits:
            if expected in ranked_folders[:k]:
                folder_hits[k] += 1

        oracle_matches = find_answer_pages(
            args.data_dir / expected,
            question["answer"],
            question.get("evidence", ""),
        )
        oracle_pages = [match.page for match in oracle_matches]
        oracle_page_keys = {f"{expected}_{page}" for page in oracle_pages}
        oracle_rank = None
        if oracle_pages:
            oracle_available += 1
            for rank, page_key in enumerate(ranked_page_keys, start=1):
                if page_key in oracle_page_keys:
                    oracle_rank = rank
                    break
            for k in oracle_hits:
                if oracle_rank is not None and oracle_rank <= k:
                    oracle_hits[k] += 1
            if folder_rank is not None and oracle_rank is None:
                folder_found_oracle_missed += 1
            if folder_rank is not None and oracle_rank is not None:
                folder_and_oracle_found += 1
        elif len(no_oracle_rows) < args.examples:
            no_oracle_rows.append(
                {
                    "question": question["question"],
                    "answer": question["answer"],
                    "expected_folder": expected,
                }
            )

        row = {
            "question": question["question"],
            "answer": question["answer"],
            "type": question["type"],
            "expected_folder": expected,
            "folder_rank": folder_rank,
            "oracle_pages": [asdict(match) for match in oracle_matches[:10]],
            "oracle_rank": oracle_rank,
            "top10_candidates": candidates[:10],
        }
        rows.append(row)
        if oracle_pages and oracle_rank is None and len(failure_examples) < args.examples:
            failure_examples.append(row)
        if oracle_rank is not None and len(success_examples) < args.examples:
            success_examples.append(row)

        print(
            f"[{qi}/{len(questions)}] folder_rank={folder_rank} oracle_rank={oracle_rank} "
            f"expected={expected} oracle_pages={oracle_pages[:5]}"
        )

    summary = {
        "total": len(rows),
        "top_k": args.top_k,
        "index_name": args.index_name,
        "model_id": args.model_id,
        "family": family,
        "manifest_validation": manifest.get("validation"),
        "index": manifest.get("index"),
        "folder_recall": recall_dict(folder_hits, len(rows)),
        "oracle_available": oracle_available,
        "oracle_available_rate": oracle_available / max(len(rows), 1),
        "oracle_page_recall": recall_dict(oracle_hits, oracle_available),
        "folder_found_oracle_missed": folder_found_oracle_missed,
        "folder_and_oracle_found": folder_and_oracle_found,
    }

    output = {
        "summary": summary,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "no_oracle_examples": no_oracle_rows,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Folder recall: {summary['folder_recall']}")
    print(f"Oracle page recall: {summary['oracle_page_recall']}")
    print(f"Oracle available: {oracle_available}/{len(rows)}")
    print(f"Folder found but oracle page missed: {folder_found_oracle_missed}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
