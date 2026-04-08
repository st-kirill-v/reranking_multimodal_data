import sys
from pathlib import Path
import json
import pickle
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def index_all_blocks():
    blocks_base = Path("/home/user-13/reranking_multimodal_data/data/blocks_pymupdf")

    # Загружаем эмбеддер для текста
    print("Loading text embedder...")
    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_texts = []
    all_metadata = []

    folder_count = 0
    for folder_num in range(0, 229):
        folder = blocks_base / str(folder_num)
        if not folder.exists():
            continue

        json_file = folder / "content.json"
        if not json_file.exists():
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            pages = json.load(f)

        for page in pages:
            content = page.get("content", "")
            if content:
                all_texts.append(content)
                all_metadata.append(
                    {"folder": folder_num, "page": page.get("page_number", 0), "type": "markdown"}
                )

        folder_count += 1
        if folder_count % 50 == 0:
            print(f"Processed {folder_count} folders")

    print(f"Total texts to index: {len(all_texts)}")

    # Создаём эмбеддинги
    print("Creating embeddings...")
    embeddings = text_embedder.encode(all_texts, show_progress_bar=True)

    # Сохраняем
    output_dir = Path("/home/user-13/reranking_multimodal_data/index_pymupdf")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    index_all_blocks()
