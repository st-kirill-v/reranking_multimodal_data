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
        object.__setattr__(self, "index_dir", (self.index_dir or root / "index").resolve())
        object.__setattr__(self, "cache_dir", (self.cache_dir or root / "cache").resolve())


@dataclass(frozen=True)
class EmbedderConfig:
    model_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    backend: str = "sentence-transformers"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 1
    normalize: bool = True
    query_prompt: str = "Represent the user's input."


@dataclass(frozen=True)
class IndexConfig:
    name: str = "pages_qwen3"
    metric: str = "ip"
    rebuild: bool = False

    @property
    def index_filename(self) -> str:
        return f"{self.name}.index"

    @property
    def metadata_filename(self) -> str:
        return f"metadata_{self.name.removeprefix('pages_')}.json"

    @property
    def manifest_filename(self) -> str:
        return f"manifest_{self.name}.json"


@dataclass(frozen=True)
class RetrievalConfig:
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
class GeneratorConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    max_pages: int = 5
    max_image_long_edge: int | None = 1600


@dataclass(frozen=True)
class PipelineConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
