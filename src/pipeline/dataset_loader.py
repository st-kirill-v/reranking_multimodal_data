import json
import os
from typing import List, Dict, Any


def load_squad_v2_local() -> Dict[str, Any]:
    """Загружаем SQuAD 2.0"""

    # Пути к файлам
    base_path = os.path.join(os.path.dirname(__file__), "../..")
    train_path = os.path.join(base_path, "data/datasets/squad/train-v2.0.json")
    dev_path = os.path.join(base_path, "data/datasets/squad/dev-v2.0.json")

    def load_json_file(file_path: str) -> List[Dict]:
        """Загружаем JSON файл"""
        print(f"📖 Чтение файла: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["data"]

    # Проверяем существование файлов
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Файл не найден: {train_path}")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Файл не найден: {dev_path}")

    # Загружаем данные
    train_data = load_json_file(train_path)
    dev_data = load_json_file(dev_path)

    # Извлекаем контексты как документы
    documents = []
    doc_ids = []

    for split_name, data in [("train", train_data), ("dev", dev_data)]:
        for article_idx, article in enumerate(data):
            for para_idx, paragraph in enumerate(article["paragraphs"]):
                context = paragraph["context"]
                documents.append(context)
                doc_ids.append(f"squad_{split_name}_{article_idx}_{para_idx}")

    print(f"✅ Загружено {len(documents)} документов из SQuAD 2.0")
    print(f"📊 Train статей: {len(train_data)}")
    print(f"📊 Dev статей: {len(dev_data)}")
    print(f"📄 Пример документа: {documents[0][:100]}...")

    return {
        "documents": documents,
        "doc_ids": doc_ids,
        "train_data": train_data,
        "dev_data": dev_data,
    }


if __name__ == "__main__":
    # Тестируем загрузку
    data = load_squad_v2_local()
    print("Загрузчик работает!")
