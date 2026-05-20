from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean self-contained DocBench folder-recall evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/docbench"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--index-name", default="pages_qwen3_docapi_clean")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument(
        "--output", type=Path, default=Path("data/retrieval_eval_qwen3_docapi_clean.json")
    )
    return parser.parse_args()


def metadata_filename(index_name: str) -> str:
    return f"metadata_{index_name.removeprefix('pages_')}.json"


def load_questions(data_dir: Path, types: set[str]) -> list[dict]:
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
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    if index.ntotal != len(records):
        raise ValueError(f"Index/metadata mismatch: {index.ntotal} != {len(records)}")

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

    hits = {1: 0, 5: 0, 10: 0, args.top_k: 0}
    rows = []
    for i, question in enumerate(questions, start=1):
        query_embedding = encode_query(model, question["question"])
        scores, indices = index.search(query_embedding, args.top_k)
        candidates = []
        ranked_folders = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            record = records[int(idx)]
            ranked_folders.append(str(record["folder"]))
            candidates.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(idx),
                    "folder": str(record["folder"]),
                    "page": int(record["page"]),
                    "path": record["path"],
                }
            )

        expected = str(question["folder"])
        folder_rank = ranked_folders.index(expected) + 1 if expected in ranked_folders else None
        for k in hits:
            if expected in ranked_folders[:k]:
                hits[k] += 1
        rows.append(
            {
                "question": question["question"],
                "answer": question["answer"],
                "type": question["type"],
                "expected_folder": expected,
                "folder_rank": folder_rank,
                "candidates": candidates,
            }
        )
        top = candidates[0] if candidates else {}
        print(
            f"[{i}/{len(questions)}] folder_rank={folder_rank} "
            f"top={top.get('folder')}/{top.get('page')} score={top.get('score')}"
        )

    total = max(len(rows), 1)
    summary = {
        "total": len(rows),
        "top_k": args.top_k,
        "index_name": args.index_name,
        "model_id": args.model_id,
        "encoding_api": "docapi",
        "manifest_validation": manifest.get("validation"),
        "folder_recall": {f"R@{k}": hits[k] / total for k in sorted(hits)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
