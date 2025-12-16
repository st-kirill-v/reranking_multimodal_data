"""
универсальный загрузчик датасетов в rag систему.
поддерживает squad, txt, json, csv, pdf форматы.
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
import requests
import pypdf


class DatasetToRAG:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.documents_endpoint = f"{api_url}/documents"
        self.info_endpoint = f"{api_url}/info"
        self.health_endpoint = f"{api_url}/health"

    def check_server(self) -> bool:
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except:
            return False

    def load_squad(self, filepath: str, max_docs: int = None) -> List[str]:
        from src.pipeline.dataset_loader import load_squad_v2_local

        squad_data = load_squad_v2_local()

        documents = []

        if isinstance(squad_data, list):
            documents = squad_data
        elif isinstance(squad_data, dict) and "documents" in squad_data:
            documents = squad_data["documents"]
        elif isinstance(squad_data, dict) and "data" in squad_data:
            for article in squad_data["data"]:
                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    if context:
                        documents.append(context)
        else:
            return []

        if max_docs and len(documents) > max_docs:
            documents = documents[:max_docs]

        return documents

    def load_txt(self, filepath: str, max_docs: int = None) -> List[str]:
        documents = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(line)

                    if max_docs and len(documents) >= max_docs:
                        break

        return documents

    def load_json(self, filepath: str, text_field: str = "text", max_docs: int = None) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    documents.append(item[text_field])
                elif isinstance(item, str):
                    documents.append(item)

                if max_docs and len(documents) >= max_docs:
                    break

        elif isinstance(data, dict) and "documents" in data:
            for doc in data["documents"]:
                if isinstance(doc, str):
                    documents.append(doc)
                elif isinstance(doc, dict) and text_field in doc:
                    documents.append(doc[text_field])

                if max_docs and len(documents) >= max_docs:
                    break

        return documents

    def load_csv(self, filepath: str, text_column: str = "text", max_docs: int = None) -> List[str]:
        documents = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row:
                    documents.append(row[text_column])

                if max_docs and len(documents) >= max_docs:
                    break

        return documents

    def load_pdf(self, filepath: str, max_docs: int = None) -> List[str]:
        documents = []

        try:
            with open(filepath, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append(f"страница {page_num + 1}: {text[:500]}...")

                    if max_docs and len(documents) >= max_docs:
                        break
        except Exception as e:
            print(f"ошибка чтения pdf: {e}")

        return documents

    def load_from_folder(
        self, folder_path: str, file_ext: str = ".txt", max_docs: int = None
    ) -> List[str]:
        documents = []

        for filename in os.listdir(folder_path):
            if filename.endswith(file_ext):
                filepath = os.path.join(folder_path, filename)

                if filename.endswith(".txt"):
                    docs = self.load_txt(filepath)
                elif filename.endswith(".json"):
                    docs = self.load_json(filepath)
                elif filename.endswith(".csv"):
                    docs = self.load_csv(filepath)
                elif filename.endswith(".pdf"):
                    docs = self.load_pdf(filepath)
                else:
                    continue

                documents.extend(docs)

                if max_docs and len(documents) >= max_docs:
                    documents = documents[:max_docs]
                    break

        return documents

    def add_to_rag(self, documents: List[str], batch_size: int = 100) -> int:
        if not documents:
            print("нет документов для добавления")
            return 0

        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            try:
                response = requests.post(
                    self.documents_endpoint, json={"documents": batch}, timeout=30
                )

                if response.status_code == 200:
                    total_added += len(batch)
                else:
                    print(f"ошибка {response.status_code}")
                    break

            except Exception as e:
                print(f"ошибка: {e}")
                break

        return total_added

    def run_interactive(self):
        print("универсальный загрузчик датасетов в rag")

        if not self.check_server():
            print("rag сервер не запущен")
            return

        print("сервер работает")

        print("\nвыберите тип датасета:")
        print("1. squad (json)")
        print("2. текстовый файл (.txt)")
        print("3. json файл")
        print("4. csv файл")
        print("5. pdf файл")
        print("6. все файлы из папки")
        print("0. выход")

        try:
            choice = int(input("ваш выбор: ").strip())
        except:
            print("неверный выбор")
            return

        if choice == 0:
            return

        if choice in [1, 2, 3, 4, 5]:
            filepath = input("введите путь к файлу: ").strip()
            if not os.path.exists(filepath):
                print(f"файл не найден: {filepath}")
                return
        elif choice == 6:
            folder_path = input("введите путь к папке: ").strip()
            if not os.path.exists(folder_path):
                print(f"папка не найдена: {folder_path}")
                return
            file_ext = input("расширение файлов (например .txt): ").strip()

        try:
            max_docs = int(input("максимальное количество документов (0 = все): ").strip())
            if max_docs <= 0:
                max_docs = None
        except:
            max_docs = None

        documents = []

        if choice == 1:
            documents = self.load_squad(filepath, max_docs)
        elif choice == 2:
            documents = self.load_txt(filepath, max_docs)
        elif choice == 3:
            text_field = input("поле с текстом (по умолчанию 'text'): ").strip() or "text"
            documents = self.load_json(filepath, text_field, max_docs)
        elif choice == 4:
            text_column = input("колонка с текстом (по умолчанию 'text'): ").strip() or "text"
            documents = self.load_csv(filepath, text_column, max_docs)
        elif choice == 5:
            documents = self.load_pdf(filepath, max_docs)
        elif choice == 6:
            documents = self.load_from_folder(folder_path, file_ext, max_docs)

        if not documents:
            print("не удалось загрузить документы")
            return

        print(f"загружено {len(documents)} документов")

        if input("добавить документы в rag? (y/n): ").lower() == "y":
            added = self.add_to_rag(documents)
            print(f"добавлено {added} документов")

            try:
                response = requests.get(self.info_endpoint)
                if response.status_code == 200:
                    info = response.json()
                    print(f"всего в системе: {info.get('total_documents', 0)} документов")
            except:
                pass


def main():
    loader = DatasetToRAG()
    loader.run_interactive()


if __name__ == "__main__":
    main()
