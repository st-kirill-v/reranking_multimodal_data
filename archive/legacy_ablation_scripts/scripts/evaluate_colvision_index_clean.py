from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a clean sharded ColPali/ColQwen visual page index."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index_colpali_v1_3"))
    parser.add_argument("--index-name", default="pages_colpali_v1_3_clean")
    parser.add_argument("--model-id", default="vidore/colpali-v1.3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/retrieval_eval_colvision_clean.json")
    )
    return parser.parse_args()


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
    model = model_cls.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
    ).eval()
    processor = processor_cls.from_pretrained(model_id)
    return model, processor, family


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
                        "type": row.get("type", ""),
                    }
                )
    return questions


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


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "mean": None, "p50": None, "p75": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


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

    recall_cutoffs = sorted({1, 5, 10, 30, args.top_k})
    recall_cutoffs = [k for k in recall_cutoffs if k <= args.top_k]
    hits = {k: 0 for k in recall_cutoffs}
    hits_by_type: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in recall_cutoffs})
    count_by_type: Counter[str] = Counter()
    top1_folder_counts: Counter[str] = Counter()
    top1_page_counts: Counter[str] = Counter()
    top1_scores = []
    found_ranks = []
    rows = []
    success_examples = []
    failure_examples = []

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
        for rank, (score, _, record) in enumerate(ranked, start=1):
            folder = str(record["folder"])
            page = int(record["page"])
            ranked_folders.append(folder)
            if rank <= 10:
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
        if folder_rank is not None:
            found_ranks.append(folder_rank)
        for k in hits:
            if expected in ranked_folders[:k]:
                hits[k] += 1
        q_type = question["type"] or "unknown"
        count_by_type[q_type] += 1
        for k in hits_by_type[q_type]:
            if expected in ranked_folders[:k]:
                hits_by_type[q_type][k] += 1

        if candidates:
            top1_folder_counts[candidates[0]["folder"]] += 1
            top1_page_counts[f"{candidates[0]['folder']}_{candidates[0]['page']}"] += 1
            top1_scores.append(float(candidates[0]["score"]))

        row = {
            "question": question["question"],
            "answer": question["answer"],
            "type": q_type,
            "expected_folder": expected,
            "folder_rank": folder_rank,
            "top10_candidates": candidates,
        }
        rows.append(row)
        if folder_rank is None and len(failure_examples) < args.examples:
            failure_examples.append(row)
        if folder_rank is not None and len(success_examples) < args.examples:
            success_examples.append(row)

        top = candidates[0] if candidates else {}
        print(
            f"[{qi}/{len(questions)}] rank={folder_rank} "
            f"expected={expected} top={top.get('folder')}/{top.get('page')} score={top.get('score')}"
        )

    total = len(rows)
    per_type = {}
    for q_type, type_hits in hits_by_type.items():
        per_type[q_type] = {
            "count": count_by_type[q_type],
            "folder_recall": recall_dict(type_hits, count_by_type[q_type]),
        }

    summary = {
        "total": total,
        "top_k": args.top_k,
        "index_name": args.index_name,
        "model_id": args.model_id,
        "family": family,
        "manifest_validation": manifest.get("validation"),
        "index": manifest.get("index"),
        "folder_recall": recall_dict(hits, total),
        "per_type": per_type,
        "rank_stats_for_found_expected_folder": {
            "count": len(found_ranks),
            "mean": float(np.mean(found_ranks)) if found_ranks else None,
            "median": float(np.median(found_ranks)) if found_ranks else None,
            "min": min(found_ranks) if found_ranks else None,
            "max": max(found_ranks) if found_ranks else None,
        },
        "score_stats": {"top1": quantiles(top1_scores)},
        "hubness": {
            "top1_folder_top10": top1_folder_counts.most_common(10),
            "top1_page_top10": top1_page_counts.most_common(10),
            "top1_folder_concentration_top10": sum(
                count for _, count in top1_folder_counts.most_common(10)
            )
            / max(total, 1),
            "top1_page_concentration_top10": sum(
                count for _, count in top1_page_counts.most_common(10)
            )
            / max(total, 1),
        },
    }

    output = {
        "summary": summary,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nINTERPRETATION")
    print(f"Folder recall: {summary['folder_recall']}")
    print(f"Hubness top folders: {summary['hubness']['top1_folder_top10'][:5]}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
