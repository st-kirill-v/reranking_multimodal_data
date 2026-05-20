from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageRecord:
    folder: str
    page: int
    path: Path
    index: int | None = None
    tile_id: str | None = None
    crop_box: tuple[int, int, int, int] | None = None
    source_page_width: int | None = None
    source_page_height: int | None = None

    @property
    def doc_id(self) -> str:
        return f"{self.folder}_{self.page}"

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PageRecord":
        return cls(
            folder=str(data["folder"]),
            page=int(data["page"]),
            path=Path(data["path"]),
            index=data.get("index"),
            tile_id=data.get("tile_id"),
            crop_box=tuple(data["crop_box"]) if data.get("crop_box") else None,
            source_page_width=data.get("source_page_width"),
            source_page_height=data.get("source_page_height"),
        )


@dataclass
class RetrievalCandidate:
    folder: str
    page: int
    path: Path
    score: float
    rank: int
    index: int
    source: str = "qwen3_page_embedding"
    rerank_score: float | None = None
    tile_id: str | None = None
    crop_box: tuple[int, int, int, int] | None = None

    @property
    def doc_id(self) -> str:
        return f"{self.folder}_{self.page}"

    def to_json(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "page": self.page,
            "path": str(self.path),
            "score": self.score,
            "rank": self.rank,
            "index": self.index,
            "source": self.source,
            "rerank_score": self.rerank_score,
            "tile_id": self.tile_id,
            "crop_box": list(self.crop_box) if self.crop_box else None,
        }

    @classmethod
    def from_page_record(
        cls, record: PageRecord, *, score: float, rank: int, index: int
    ) -> "RetrievalCandidate":
        return cls(
            folder=record.folder,
            page=record.page,
            path=record.path,
            score=score,
            rank=rank,
            index=index,
            tile_id=record.tile_id,
            crop_box=record.crop_box,
        )
