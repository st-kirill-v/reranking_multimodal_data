from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import (
    EmbedderConfig,
    GeneratorConfig,
    IndexConfig,
    PipelineConfig,
    ProjectPaths,
    RerankerConfig,
    RetrievalConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean multimodal page-RAG pipeline.")
    parser.add_argument("query")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-name", default="pages_qwen3")
    parser.add_argument("--embed-device", default="cuda")
    parser.add_argument("--rerank-device", default="cuda")
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--first-stage-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-generator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.mmrag.pipeline import PageRAGPipeline

    config = PipelineConfig(
        paths=ProjectPaths(data_dir=args.data_dir, index_dir=args.index_dir),
        embedder=EmbedderConfig(device=args.embed_device),
        index=IndexConfig(name=args.index_name),
        retrieval=RetrievalConfig(
            first_stage_top_k=args.first_stage_top_k,
            rerank_top_k=args.rerank_top_k,
            final_top_k=args.final_top_k,
            neighbor_radius=args.neighbor_radius,
        ),
        reranker=RerankerConfig(device=args.rerank_device),
        generator=GeneratorConfig(device=args.generator_device, max_pages=args.final_top_k),
    )
    pipeline = PageRAGPipeline(
        config,
        use_reranker=not args.no_reranker,
        use_generator=not args.no_generator,
    )
    result = pipeline.answer(args.query)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
