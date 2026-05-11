import sys
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModel, AutoProcessor


def load_texts_from_docling(docling_dir: Path):
    for folder_num in range(0, 229):
        folder = docling_dir / str(folder_num)
        md_file = folder / "content.md"
        if not md_file.exists():
            continue
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        if content:
            yield content, {"folder": folder_num, "type": "text", "source": "docling"}


def encode_texts_sequential(texts_generator, model, processor, device, batch_size=4):
    all_embeds = []
    all_metadata = []
    batch_texts = []
    batch_meta = []

    for text, meta in tqdm(texts_generator, desc="Encoding texts", total=229):
        batch_texts.append(text)
        batch_meta.append(meta)

        if len(batch_texts) >= batch_size:
            inputs = processor(
                text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeds = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
            all_embeds.extend(embeds)
            all_metadata.extend(batch_meta)

            batch_texts = []
            batch_meta = []
            del inputs, outputs
            torch.cuda.empty_cache()

    if batch_texts:
        inputs = processor(
            text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
        all_embeds.extend(embeds)
        all_metadata.extend(batch_meta)

    return np.array(all_embeds), all_metadata


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
    output_dir = Path("/home/user-13/reranking_multimodal_data/index_text_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading and encoding texts from Docling (all 229 folders)...")
    texts_gen = load_texts_from_docling(docling_dir)
    text_embeds, text_metadata = encode_texts_sequential(texts_gen, model, processor, device)
    print(f"Text embeddings shape: {text_embeds.shape}")
    print(f"Total text chunks: {len(text_metadata)}")

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(text_embeds.shape[1])
    index.add(text_embeds.astype(np.float32))

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(text_metadata, f)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
