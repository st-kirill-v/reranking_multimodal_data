from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import EmbedderConfig, IndexConfig, PipelineConfig, ProjectPaths
from src.mmrag.dataset import load_docbench_questions
from src.mmrag.indexing import FaissPageIndex
from src.mmrag.oracle import find_answer_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank pages only within the expected document folder."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding-api", choices=["legacy_encode", "docapi"], default="docapi")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/audit_in_document_ranking.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.embeddings import Qwen3PageEmbedder

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(
            model_id=args.model_id,
            device=args.device,
            encoding_api=args.encoding_api,
        ),
        index=IndexConfig(name=args.index_name),
    )
    questions = load_docbench_questions(config.paths.data_dir, question_types=set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    page_index = FaissPageIndex(config).load()
    vectors = page_index.vectors()
    embedder = Qwen3PageEmbedder(config.embedder)

    rows = []
    oracle_available = 0
    page_hits_at_1 = 0
    page_hits_at_5 = 0
    page_hits_any = 0

    for i, q in enumerate(questions, start=1):
        expected_folder = str(q["folder"])
        oracle_matches = find_answer_pages(
            config.paths.data_dir / expected_folder,
            q["answer"],
            q.get("evidence", ""),
        )
        oracle_pages = [match.page for match in oracle_matches[:5]]
        folder_indices = [
            idx for idx, record in enumerate(page_index.records) if record.folder == expected_folder
        ]
        query_embedding = embedder.encode_query(q["question"])[0]
        scored = []
        for idx in folder_indices:
            record = page_index.records[idx]
            scored.append((float(np.dot(query_embedding, vectors[idx])), record.page, idx))
        scored.sort(reverse=True)
        ranked_pages = []
        seen_pages = set()
        for score, page, idx in scored:
            if page in seen_pages:
                continue
            seen_pages.add(page)
            ranked_pages.append({"page": page, "score": score, "index": idx})

        oracle_rank = None
        if oracle_pages:
            oracle_available += 1
            for rank, item in enumerate(ranked_pages, start=1):
                if item["page"] in oracle_pages:
                    oracle_rank = rank
                    break
            page_hits_at_1 += 1 if oracle_rank == 1 else 0
            page_hits_at_5 += 1 if oracle_rank is not None and oracle_rank <= 5 else 0
            page_hits_any += 1 if oracle_rank is not None else 0

        row = {
            "folder": expected_folder,
            "question": q["question"],
            "answer": q["answer"],
            "oracle_pages": oracle_pages,
            "oracle_rank_within_doc": oracle_rank,
            "top_pages_within_doc": ranked_pages[:10],
        }
        rows.append(row)
        print(
            f"[{i}/{len(questions)}] folder={expected_folder} oracle_pages={oracle_pages} "
            f"oracle_rank={oracle_rank} top_page={ranked_pages[0]['page'] if ranked_pages else None}"
        )

    denom = max(oracle_available, 1)
    summary = {
        "questions": len(rows),
        "oracle_available": oracle_available,
        "page_recall_at_1_within_doc": page_hits_at_1 / denom,
        "page_recall_at_5_within_doc": page_hits_at_5 / denom,
        "page_recall_any_within_doc": page_hits_any / denom,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
