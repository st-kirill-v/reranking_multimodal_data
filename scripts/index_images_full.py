import sys
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModel, AutoProcessor


def load_images_from_pymupdf(pymupdf_dir: Path):
    for folder_num in range(0, 229):
        folder = pymupdf_dir / str(folder_num)
        images_dir = folder / "images"
        if not images_dir.exists():
            continue
        for img_path in images_dir.glob("*.png"):
            yield str(img_path), {
                "folder": folder_num,
                "type": "image",
                "source": "pymupdf",
                "path": str(img_path),
            }


def encode_images_sequential(images_generator, model, processor, device, batch_size=4):
    all_embeds = []
    all_metadata = []
    batch_paths = []
    batch_meta = []

    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    image_pad = "<|image_pad|>"
    image_token = f"{vision_start}{image_pad}{vision_end}"

    for img_path, meta in tqdm(images_generator, desc="Encoding images"):
        batch_paths.append(img_path)
        batch_meta.append(meta)

        if len(batch_paths) >= batch_size:
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))

            texts = [image_token] * len(images)

            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
                device
            )

            with torch.no_grad():
                outputs = model(**inputs)
                embeds = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()

            all_embeds.extend(embeds)
            all_metadata.extend(batch_meta)

            batch_paths = []
            batch_meta = []
            del images, texts, inputs, outputs
            torch.cuda.empty_cache()

    if batch_paths:
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))

        texts = [image_token] * len(images)

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

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

    pymupdf_dir = Path("/home/user-13/reranking_multimodal_data/data/blocks_pymupdf")
    output_dir = Path("/home/user-13/reranking_multimodal_data/index_images_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading and encoding images from PyMuPDF4LLM (all 229 folders)...")
    images_gen = load_images_from_pymupdf(pymupdf_dir)
    image_embeds, image_metadata = encode_images_sequential(images_gen, model, processor, device)
    print(f"Image embeddings shape: {image_embeds.shape}")
    print(f"Total images: {len(image_metadata)}")

    if len(image_embeds) == 0:
        print("No images found to index!")
        return

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(image_embeds.shape[1])
    index.add(image_embeds.astype(np.float32))

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(image_metadata, f)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
