from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = field(default_factory=find_project_root)
    data_dir: Path | None = None
    index_dir: Path | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        root = self.root.resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self,
            "data_dir",
            (self.data_dir or root / "data" / "datasets" / "docbench").resolve(),
        )
        object.__setattr__(
            self,
            "index_dir",
            (self.index_dir or root / "index_colpali_v1_3_merged").resolve(),
        )
        object.__setattr__(self, "cache_dir", (self.cache_dir or root / "cache").resolve())


@dataclass(frozen=True)
class RetrievalConfig:
    model_id: str = "vidore/colpali-v1.3-merged"
    index_name: str = "pages_colpali_v1_3_merged_clean"
    first_stage_top_k: int = 30
    rerank_top_k: int = 10
    final_top_k: int = 5
    neighbor_radius: int = 0


@dataclass(frozen=True)
class RerankerConfig:
    model_id: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 1
    max_input_tiles: int = 6
    use_thumbnail: bool = True
    max_length: int = 2048


@dataclass(frozen=True)
class PipelineConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
