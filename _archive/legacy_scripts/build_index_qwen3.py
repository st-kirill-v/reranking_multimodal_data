import torch
from transformers import AutoModel, AutoProcessor
from pathlib import Path
import faiss
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import time
import os

os.environ["TORCH_LOAD_META_DEVICE"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

data_path = Path("/home/user-13/reranking_multimodal_data/data/datasets/docbench")
index_dir = Path("/home/user-13/reranking_multimodal_data/index")
index_dir.mkdir(exist_ok=True)

print("\nLoading Qwen3-VL-Embedding-8B...")
start_time = time.time()

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-8B", trust_remote_code=True)

print(f"Model loaded in {time.time() - start_time:.2f}s")

print("\nCollecting pages from all folders...")
pages = []

doc_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and f.name.isdigit()])

for folder in tqdm(doc_folders, desc="Scanning folders"):
    pages_dir = folder / "extracted" / "pages"
    if not pages_dir.exists():
        print(f"  Warning: {pages_dir} not found")
        continue

    img_files = list(pages_dir.glob("*.png"))
    if not img_files:
        print(f"  Warning: No PNG in {pages_dir}")
        continue

    for img_path in sorted(img_files):
        try:
            stem = img_path.stem
            if stem.startswith("page_"):
                page_num = int(stem.split("_")[1])
            else:
                page_num = int(stem)
        except:
            print(f"  Warning: Could not parse page number from {img_path.name}")
            continue

        pages.append({"folder": folder.name, "page": page_num, "path": str(img_path)})

print(f"Total pages found: {len(pages)}")
if not pages:
    raise RuntimeError("No pages found")

embed_dim = 4096
index = faiss.IndexFlatIP(embed_dim)

print("\nEncoding pages with Qwen3-VL-Embedding...")
batch_size = 4
all_embeddings = []
metadata = []

total_batches = (len(pages) + batch_size - 1) // batch_size
pbar = tqdm(total=total_batches, desc="Encoding batches", unit="batch")

for i in range(0, len(pages), batch_size):
    batch = pages[i : i + batch_size]
    images = []
    valid_indices = []

    for j, p in enumerate(batch):
        try:
            img = Image.open(p["path"]).convert("RGB")
            images.append(img)
            valid_indices.append(j)
        except Exception as e:
            print(f"Warning: Could not load {p['path']}: {e}")
            continue

    if not images:
        pbar.update(1)
        continue

    with torch.no_grad():
        texts = ["<|image_pad|>"] * len(images)

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)

        image_embeds = outputs.last_hidden_state.mean(dim=1)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        embs = image_embeds

    all_embeddings.append(embs.cpu().float().numpy())

    for idx_in_batch, orig_idx in enumerate(valid_indices):
        p = batch[orig_idx]
        metadata.append(
            {"folder": p["folder"], "page": p["page"], "path": p["path"], "index": len(metadata)}
        )

    pbar.update(1)
    pbar.set_postfix({"batch": f"{len(images)}/{batch_size}", "processed": len(metadata)})

pbar.close()

all_embeddings = np.vstack(all_embeddings)
index.add(all_embeddings)
faiss.write_index(index, str(index_dir / "pages_qwen3.index"))

with open(index_dir / "metadata_qwen3.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nQwen3-VL index saved to {index_dir / 'pages_qwen3.index'}")
print(f"Total vectors: {len(metadata)}")
print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
