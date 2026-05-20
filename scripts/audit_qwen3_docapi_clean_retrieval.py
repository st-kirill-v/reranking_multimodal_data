from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Qwen3 docapi retrieval audit: recall, hubs, scores, per-type breakdown."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="pages_qwen3_docapi_clean")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit_qwen3_docapi_clean_retrieval.json")
    )
    return parser.parse_args()


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('pages_')}.json"


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


def encode_query(model: Any, query: str) -> np.ndarray:
    embedding = model.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray([embedding], dtype=np.float32)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "mean": None, "p50": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def update_hits(hits: dict[int, int], ranked_folders: list[str], expected: str) -> None:
    for k in hits:
        if expected in ranked_folders[:k]:
            hits[k] += 1


def recall_dict(hits: dict[int, int], total: int) -> dict[str, float]:
    denominator = max(total, 1)
    return {f"R@{k}": hits[k] / denominator for k in sorted(hits)}


def main() -> None:
    args = parse_args()
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    index_path = args.index_dir / f"{args.index_name}.index"
    metadata_path = args.index_dir / metadata_filename(args.index_name)
    manifest_path = args.index_dir / f"manifest_{args.index_name}.json"

    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if index.ntotal != len(records):
        raise ValueError(f"Index/metadata mismatch: {index.ntotal} != {len(records)}")

    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    model_kwargs = {"dtype": dtype} if dtype is not None else {}
    model = SentenceTransformer(
        args.model_id,
        trust_remote_code=True,
        device=args.device,
        model_kwargs=model_kwargs,
    )

    questions = load_questions(args.data_dir, set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    recall_cutoffs = sorted({1, 5, 10, 30, 100, args.top_k})
    recall_cutoffs = [k for k in recall_cutoffs if k <= args.top_k]
    hits = {k: 0 for k in recall_cutoffs}
    hits_by_type: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in recall_cutoffs})
    count_by_type: Counter[str] = Counter()

    top1_folder_counts: Counter[str] = Counter()
    top1_page_counts: Counter[str] = Counter()
    topk_folder_counts: Counter[str] = Counter()
    topk_page_counts: Counter[str] = Counter()

    top1_scores: list[float] = []
    expected_folder_scores: list[float] = []
    expected_folder_ranks: list[int] = []
    miss_top1_scores: list[float] = []
    hit_top1_scores: list[float] = []

    rows = []
    successes = []
    failures = []

    for i, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, question["question"])
        scores, indices = index.search(query_embedding, args.top_k)
        candidates = []
        ranked_folders = []
        expected = str(question["folder"])
        expected_best_score = None

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            record = records[int(idx)]
            folder = str(record["folder"])
            page = int(record["page"])
            page_key = f"{folder}_{page}"
            score_value = float(score)
            ranked_folders.append(folder)
            if rank == 1:
                top1_folder_counts[folder] += 1
                top1_page_counts[page_key] += 1
                top1_scores.append(score_value)
            topk_folder_counts[folder] += 1
            topk_page_counts[page_key] += 1
            if folder == expected and expected_best_score is None:
                expected_best_score = score_value
            if rank <= 10:
                candidates.append(
                    {
                        "rank": rank,
                        "score": score_value,
                        "index": int(idx),
                        "folder": folder,
                        "page": page,
                        "path": record["path"],
                    }
                )

        folder_rank = ranked_folders.index(expected) + 1 if expected in ranked_folders else None
        update_hits(hits, ranked_folders, expected)
        q_type = question["type"] or "unknown"
        count_by_type[q_type] += 1
        update_hits(hits_by_type[q_type], ranked_folders, expected)

        if folder_rank is not None:
            expected_folder_ranks.append(folder_rank)
            if expected_best_score is not None:
                expected_folder_scores.append(expected_best_score)

        top1_score = candidates[0]["score"] if candidates else None
        if folder_rank == 1 and top1_score is not None:
            hit_top1_scores.append(top1_score)
        elif top1_score is not None:
            miss_top1_scores.append(top1_score)

        row = {
            "question": question["question"],
            "answer": question["answer"],
            "type": q_type,
            "expected_folder": expected,
            "folder_rank": folder_rank,
            "top10_candidates": candidates,
        }
        rows.append(row)

        if folder_rank is None and len(failures) < args.examples:
            failures.append(row)
        if folder_rank is not None and len(successes) < args.examples:
            successes.append(row)

        top = candidates[0] if candidates else {}
        print(
            f"[{i}/{len(questions)}] rank={folder_rank} "
            f"expected={expected} top={top.get('folder')}/{top.get('page')} "
            f"score={top.get('score')}"
        )

    total = len(rows)
    top1_concentration_10 = sum(count for _, count in top1_folder_counts.most_common(10)) / max(
        total, 1
    )
    top1_page_concentration_10 = sum(count for _, count in top1_page_counts.most_common(10)) / max(
        total, 1
    )
    topk_total = max(total * args.top_k, 1)

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
        "encoding_api": "docapi",
        "manifest_validation": manifest.get("validation"),
        "folder_recall": recall_dict(hits, total),
        "per_type": per_type,
        "rank_stats_for_found_expected_folder": {
            "count": len(expected_folder_ranks),
            "mean": mean(expected_folder_ranks) if expected_folder_ranks else None,
            "median": float(np.median(expected_folder_ranks)) if expected_folder_ranks else None,
            "min": min(expected_folder_ranks) if expected_folder_ranks else None,
            "max": max(expected_folder_ranks) if expected_folder_ranks else None,
        },
        "score_stats": {
            "top1": quantiles(top1_scores),
            "top1_when_correct": quantiles(hit_top1_scores),
            "top1_when_wrong": quantiles(miss_top1_scores),
            "best_expected_folder_score_when_found": quantiles(expected_folder_scores),
        },
        "hubness": {
            "top1_folder_top10": top1_folder_counts.most_common(10),
            "top1_page_top10": top1_page_counts.most_common(10),
            "topk_folder_top10": topk_folder_counts.most_common(10),
            "topk_page_top10": topk_page_counts.most_common(10),
            "top1_folder_concentration_top10": top1_concentration_10,
            "top1_page_concentration_top10": top1_page_concentration_10,
            "topk_folder_share_top10": sum(count for _, count in topk_folder_counts.most_common(10))
            / topk_total,
            "topk_page_share_top10": sum(count for _, count in topk_page_counts.most_common(10))
            / topk_total,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary,
                "success_examples": successes,
                "failure_examples": failures,
                "rows": rows,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
