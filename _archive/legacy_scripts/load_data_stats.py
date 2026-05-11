import sys
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModel, AutoProcessor


def load_texts_from_docling(docling_dir: Path):
    """Загружает все Markdown тексты из Docling"""
    texts = []
    metadata = []

    for folder_num in range(0, 229):
        folder = docling_dir / str(folder_num)
        md_file = folder / "content.md"
        if not md_file.exists():
            continue

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        if content:
            texts.append(content)
            metadata.append({"folder": folder_num, "type": "markdown", "source": "docling"})

    return texts, metadata


def load_images_from_pymupdf(pymupdf_dir: Path):
    """Загружает все PNG изображения из PyMuPDF4LLM"""
    images = []
    metadata = []

    for folder_num in range(0, 229):
        folder = pymupdf_dir / str(folder_num)
        images_dir = folder / "images"
        if not images_dir.exists():
            continue

        for img_path in images_dir.glob("*.png"):
            images.append(str(img_path))
            metadata.append(
                {"folder": folder_num, "type": "image", "source": "pymupdf", "path": str(img_path)}
            )

    return images, metadata


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Qwen3-VL-Embedding...")
    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-VL-Embedding-8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True)

    docling_dir = Path("/home/user-13/reranking_multimodal_data/data/docling_markdown_fast")
    pymupdf_dir = Path("/home/user-13/reranking_multimodal_data/data/blocks_pymupdf")
    output_dir = Path("/home/user-13/reranking_multimodal_data/index_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading texts from Docling...")
    texts, text_metadata = load_texts_from_docling(docling_dir)
    print(f"Loaded {len(texts)} text chunks")

    print("Loading images from PyMuPDF4LLM...")
    images, image_metadata = load_images_from_pymupdf(pymupdf_dir)
    print(f"Loaded {len(images)} images")

    # TODO: эмбеддинги текста и изображений
    # TODO: сохранить в FAISS

    print(f"Texts: {len(texts)}")
    print(f"Images: {len(images)}")


if __name__ == "__main__":
    main()
