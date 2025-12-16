"""
загрузчик датасета squad v2.0 из локальных файлов.
"""

import json
import os
from typing import List, Dict, Any


def load_squad_v2_local() -> Dict[str, Any]:
    base_path = os.path.join(os.path.dirname(__file__), "../..")
    train_path = os.path.join(base_path, "data/datasets/squad/train-v2.0.json")
    dev_path = os.path.join(base_path, "data/datasets/squad/dev-v2.0.json")

    def load_json_file(file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["data"]

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"файл не найден: {train_path}")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"файл не найден: {dev_path}")

    train_data = load_json_file(train_path)
    dev_data = load_json_file(dev_path)

    documents = []
    doc_ids = []

    for split_name, data in [("train", train_data), ("dev", dev_data)]:
        for article_idx, article in enumerate(data):
            for para_idx, paragraph in enumerate(article["paragraphs"]):
                context = paragraph["context"]
                documents.append(context)
                doc_ids.append(f"squad_{split_name}_{article_idx}_{para_idx}")

    return {
        "documents": documents,
        "doc_ids": doc_ids,
        "train_data": train_data,
        "dev_data": dev_data,
    }


if __name__ == "__main__":
    data = load_squad_v2_local()
    print("загрузчик работает")
