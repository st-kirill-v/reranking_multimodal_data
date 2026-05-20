from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate document-level aggregation over a clean Qwen3 page index."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="pages_qwen3_docapi_clean")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--score-modes",
        nargs="*",
        default=["centroid", "max_page", "top5_mean"],
        choices=["centroid", "max_page", "top5_mean"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit_qwen3_doc_aggregation_clean.json")
    )
    return parser.parse_args()


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('pages_')}.json"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


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
    return np.asarray(embedding, dtype=np.float32)


def recall_dict(hits: dict[int, int], total: int) -> dict[str, float]:
    denominator = max(total, 1)
    return {f"R@{k}": hits[k] / denominator for k in sorted(hits)}


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


def make_interpretation(summaries: dict, *, top_k: int) -> dict:
    if not summaries:
        return {
            "verdict": "no_results",
            "message": "No aggregation modes were evaluated.",
            "recommended_next_step": "rerun_audit",
        }

    def recall_at(mode_summary: dict, k: int) -> float:
        return float(mode_summary.get("folder_recall", {}).get(f"R@{k}", 0.0))

    best_mode = max(summaries, key=lambda mode: recall_at(summaries[mode], top_k))
    best_summary = summaries[best_mode]
    best_recall = recall_at(best_summary, top_k)
    best_r30 = recall_at(best_summary, 30)
    best_r10 = recall_at(best_summary, 10)
    best_r1 = recall_at(best_summary, 1)
    concentration = float(best_summary.get("top1_folder_concentration_top10") or 0.0)
    found_count = int(
        best_summary.get("rank_stats_for_found_expected_folder", {}).get("count") or 0
    )
    total = int(best_summary.get("total") or 0)

    if best_recall < 0.4:
        verdict = "aggregation_does_not_fix_global_routing"
        message = (
            "Document-level aggregation still misses most target documents. "
            "The full-page embedding space does not contain enough query-to-document signal."
        )
        recommended = "build_and_evaluate_tile_index"
    elif best_r30 < 0.3 and best_recall >= 0.4:
        verdict = "aggregation_finds_some_targets_but_ranking_is_weak"
        message = "The expected document appears in the wider candidate pool, but ranks too low for practical RAG."
        recommended = "try_document_reranking_or_tile_index"
    else:
        verdict = "aggregation_is_promising"
        message = "Document aggregation improves routing enough to be considered as a first-stage document retriever."
        recommended = "combine_document_routing_with_page_or_tile_reranking"

    if concentration > 0.8:
        hubness_note = "Severe hubness remains: top-1 results are concentrated in a few documents."
    elif concentration > 0.5:
        hubness_note = "Moderate hubness remains in document-level routing."
    else:
        hubness_note = "Hubness is less dominant at document level."

    return {
        "verdict": verdict,
        "best_mode_by_R@top_k": best_mode,
        "best_mode_recall": {
            "R@1": best_r1,
            "R@10": best_r10,
            "R@30": best_r30,
            f"R@{top_k}": best_recall,
        },
        "found_expected_documents": found_count,
        "total_questions": total,
        "top1_folder_concentration_top10": concentration,
        "message": message,
        "hubness_note": hubness_note,
        "recommended_next_step": recommended,
    }


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

    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)

    folder_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        folder_to_indices[str(record["folder"])].append(idx)
    folders = sorted(folder_to_indices, key=lambda value: int(value) if value.isdigit() else value)

    centroids = []
    for folder in folders:
        centroids.append(np.mean(vectors[folder_to_indices[folder]], axis=0))
    centroids = l2_normalize(np.asarray(centroids, dtype=np.float32))

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

    doc_count = len(folders)
    recall_cutoffs = sorted({1, 5, 10, 30, args.top_k})
    recall_cutoffs = [k for k in recall_cutoffs if k <= min(args.top_k, doc_count)]

    summaries = {}
    all_rows = {}
    for mode in args.score_modes:
        hits = {k: 0 for k in recall_cutoffs}
        hits_by_type: dict[str, dict[int, int]] = defaultdict(
            lambda: {k: 0 for k in recall_cutoffs}
        )
        count_by_type: Counter[str] = Counter()
        top1_folders: Counter[str] = Counter()
        found_ranks = []
        expected_scores = []
        rows = []
        success_examples = []
        failure_examples = []

        for i, question in enumerate(questions, start=1):
            query = encode_query(model, question["question"])
            if mode == "centroid":
                scores = centroids @ query
            else:
                doc_scores = []
                for folder in folders:
                    page_scores = vectors[folder_to_indices[folder]] @ query
                    if mode == "max_page":
                        doc_scores.append(float(np.max(page_scores)))
                    else:
                        top = np.sort(page_scores)[-min(5, len(page_scores)) :]
                        doc_scores.append(float(np.mean(top)))
                scores = np.asarray(doc_scores, dtype=np.float32)

            order = np.argsort(-scores)
            ranked_folders = [folders[int(pos)] for pos in order[: args.top_k]]
            expected = str(question["folder"])
            folder_rank = ranked_folders.index(expected) + 1 if expected in ranked_folders else None
            if folder_rank is not None:
                found_ranks.append(folder_rank)
                expected_pos = folders.index(expected)
                expected_scores.append(float(scores[expected_pos]))
            if ranked_folders:
                top1_folders[ranked_folders[0]] += 1
            for k in hits:
                if expected in ranked_folders[:k]:
                    hits[k] += 1

            q_type = question["type"] or "unknown"
            count_by_type[q_type] += 1
            for k in hits_by_type[q_type]:
                if expected in ranked_folders[:k]:
                    hits_by_type[q_type][k] += 1

            top_docs = [
                {"rank": rank, "folder": folders[int(pos)], "score": float(scores[int(pos)])}
                for rank, pos in enumerate(order[:10], start=1)
            ]
            row = {
                "question": question["question"],
                "answer": question["answer"],
                "type": q_type,
                "expected_folder": expected,
                "folder_rank": folder_rank,
                "top10_documents": top_docs,
            }
            rows.append(row)
            if folder_rank is None and len(failure_examples) < args.examples:
                failure_examples.append(row)
            if folder_rank is not None and len(success_examples) < args.examples:
                success_examples.append(row)

            print(
                f"[{mode} {i}/{len(questions)}] rank={folder_rank} "
                f"expected={expected} top={top_docs[0]['folder'] if top_docs else None}"
            )

        per_type = {}
        for q_type, type_hits in hits_by_type.items():
            per_type[q_type] = {
                "count": count_by_type[q_type],
                "folder_recall": recall_dict(type_hits, count_by_type[q_type]),
            }

        summaries[mode] = {
            "total": len(rows),
            "top_k": args.top_k,
            "folder_recall": recall_dict(hits, len(rows)),
            "per_type": per_type,
            "rank_stats_for_found_expected_folder": {
                "count": len(found_ranks),
                "mean": float(np.mean(found_ranks)) if found_ranks else None,
                "median": float(np.median(found_ranks)) if found_ranks else None,
                "min": min(found_ranks) if found_ranks else None,
                "max": max(found_ranks) if found_ranks else None,
            },
            "expected_folder_score_when_found": quantiles(expected_scores),
            "top1_folder_top10": top1_folders.most_common(10),
            "top1_folder_concentration_top10": sum(
                count for _, count in top1_folders.most_common(10)
            )
            / max(len(rows), 1),
            "success_examples": success_examples,
            "failure_examples": failure_examples,
        }
        all_rows[mode] = rows

    interpretation = make_interpretation(summaries, top_k=args.top_k)

    output = {
        "index_name": args.index_name,
        "model_id": args.model_id,
        "encoding_api": "docapi",
        "manifest_validation": manifest.get("validation"),
        "documents": doc_count,
        "score_modes": args.score_modes,
        "interpretation": interpretation,
        "summary_by_mode": summaries,
        "rows_by_mode": all_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {k: v for k, v in output.items() if k != "rows_by_mode"}, indent=2, ensure_ascii=False
        )
    )
    print("\nINTERPRETATION")
    print(f"Verdict: {interpretation['verdict']}")
    print(f"Best mode: {interpretation['best_mode_by_R@top_k']}")
    print(f"Recall: {interpretation['best_mode_recall']}")
    print(f"Message: {interpretation['message']}")
    print(f"Hubness: {interpretation['hubness_note']}")
    print(f"Next: {interpretation['recommended_next_step']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
