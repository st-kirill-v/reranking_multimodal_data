import torch
from pathlib import Path
import faiss
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================


data_path = Path("/home/user-13/reranking_multimodal_data/data/datasets/docbench")
index_dir = Path("/home/user-13/reranking_multimodal_data/index")

index_dir.mkdir(exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Qwen3-VL-Embedding-2B...")

start_time = time.time()

model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B",
    trust_remote_code=True,
    device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
)


print(f"Loaded in {time.time() - start_time:.2f}s")

# ============================================================
# COLLECT PAGES
# ============================================================

pages = []

doc_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and f.name.isdigit()])

for folder in tqdm(doc_folders, desc="Scanning folders"):

    pages_dir = folder / "extracted" / "pages"

    if not pages_dir.exists():
        continue

    for img_path in sorted(pages_dir.glob("*.png")):

        try:

            stem = img_path.stem

            if stem.startswith("page_"):
                page_num = int(stem.split("_")[1])
            else:
                page_num = int(stem)

            pages.append(
                {
                    "folder": folder.name,
                    "page": page_num,
                    "path": str(img_path),
                }
            )

        except:
            continue

print(f"Total pages: {len(pages)}")

test_img = Image.new("RGB", (224, 224))
test_emb = model.encode([test_img], normalize_embeddings=True)
embedding_dim = test_emb.shape[1]
print(f"Real embedding dimension: {test_emb.shape[1]}")

index = faiss.IndexFlatIP(embedding_dim)

print("FAISS index initialized")

# ============================================================
# ENCODE
# ============================================================
print("\nMoving model to GPU for encoding...")

torch.cuda.empty_cache()


metadata = []

batch_size = 1

print(f"\nEncoding {len(pages)} pages...")

for i in tqdm(range(0, len(pages), batch_size), desc="Encoding"):

    batch = pages[i : i + batch_size]

    images = []
    batch_meta = []

    for p in batch:

        try:

            img = Image.open(p["path"]).convert("RGB")

            images.append(img)

            batch_meta.append(p)

        except Exception as e:

            print(f"Failed loading {p['path']}: {e}")

    if not images:
        continue

    # ========================================================
    # OFFICIAL ENCODE
    # ========================================================

    try:

        with torch.no_grad():

            embeddings = model.encode(
                images,
                normalize_embeddings=True,
                batch_size=1,
                convert_to_numpy=True,
            )

    except torch.cuda.OutOfMemoryError:

        print("OOM on batch, skipping...")
        torch.cuda.empty_cache()
        continue

    embeddings = embeddings.astype("float32")

    index.add(embeddings)

    torch.cuda.empty_cache()

    for p in batch_meta:

        metadata.append(
            {
                "folder": p["folder"],
                "page": p["page"],
                "path": p["path"],
                "index": len(metadata),
            }
        )

    current = len(metadata)
    total = len(pages)
    print(f"   [{current}/{total}] pages indexed | Last: folder={p['folder']}, page={p['page']}")

    # Каждые 100 страниц — статистика
    if current % 100 == 0:
        elapsed = time.time() - start_time
        speed = current / elapsed
        print(
            f"   {current}/{total} ({current/total*100:.1f}%) | Speed: {speed:.2f} img/sec | Time: {elapsed:.1f}s"
        )


# ============================================================
# SAVE
# ============================================================

faiss.write_index(index, str(index_dir / "pages_qwen3.index"))

with open(index_dir / "metadata_qwen3.json", "w", encoding="utf-8") as f:

    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Index saved.")
print(f"Total vectors: {len(metadata)}")
