"""
dataset_to_rag.py - УНИВЕРСАЛЬНЫЙ загрузчик датасетов в RAG
Поддерживает: SQuAD, WikiQA, CSV, TXT, JSON, PDF
"""

import sys
import os
import requests
import json
import csv
from typing import List, Dict, Any, Optional
import pypdf


class DatasetToRAG:
    """Универсальный загрузчик датасетов в RAG систему"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.documents_endpoint = f"{api_url}/documents"
        self.info_endpoint = f"{api_url}/info"
        self.health_endpoint = f"{api_url}/health"

    def check_server(self) -> bool:
        """Проверить доступность RAG сервера"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except:
            return False

    # ========== МЕТОДЫ ЗАГРУЗКИ РАЗНЫХ ФОРМАТОВ ==========

    def load_squad(self, filepath: str, max_docs: int = None) -> List[str]:
        """Загрузить SQuAD датасет"""
        from src.pipeline.dataset_loader import load_squad_v2_local

        print(f"📥 Загружаю SQuAD: {filepath}")

        # Получаем данные
        squad_data = load_squad_v2_local()

        # Если это уже список документов
        if isinstance(squad_data, list):
            documents = squad_data
        # Если это словарь с ключом "documents"
        elif isinstance(squad_data, dict) and "documents" in squad_data:
            documents = squad_data["documents"]
        # Если это старый формат SQuAD
        elif isinstance(squad_data, dict) and "data" in squad_data:
            documents = []
            for article in squad_data["data"]:
                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    if context:
                        documents.append(context)
        else:
            print("❌ Неизвестный формат данных")
            return []

        # Ограничиваем количество
        if max_docs and len(documents) > max_docs:
            documents = documents[:max_docs]

        print(f"✅ Загружено {len(documents)} документов")
        return documents

    def load_txt(self, filepath: str, max_docs: int = None) -> List[str]:
        """Загрузить текстовый файл (каждая строка = документ)"""
        documents = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    documents.append(line)

                    if max_docs and len(documents) >= max_docs:
                        break

        return documents

    def load_json(self, filepath: str, text_field: str = "text", max_docs: int = None) -> List[str]:
        """Загрузить JSON файл"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        # Если это список
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    documents.append(item[text_field])
                elif isinstance(item, str):
                    documents.append(item)

                if max_docs and len(documents) >= max_docs:
                    break

        # Если это словарь с ключом "documents"
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
        """Загрузить CSV файл"""
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
        """Загрузить PDF файл (каждая страница = документ)"""
        documents = []

        try:
            with open(filepath, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append(f"Страница {page_num + 1}: {text[:500]}...")

                    if max_docs and len(documents) >= max_docs:
                        break
        except Exception as e:
            print(f"❌ Ошибка чтения PDF: {e}")

        return documents

    def load_from_folder(
        self, folder_path: str, file_ext: str = ".txt", max_docs: int = None
    ) -> List[str]:
        """Загрузить все файлы из папки"""
        documents = []

        for filename in os.listdir(folder_path):
            if filename.endswith(file_ext):
                filepath = os.path.join(folder_path, filename)

                # Выбираем loader по расширению
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

    # ========== ОБЩИЕ МЕТОДЫ ==========

    def add_to_rag(self, documents: List[str], batch_size: int = 100) -> int:
        """Добавить документы в RAG систему"""
        if not documents:
            print("⚠️  Нет документов для добавления")
            return 0

        print(f"📤 Загружаю {len(documents)} документов...")

        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size

            print(f"   Батч {batch_num}/{total_batches}: {len(batch)} док.")

            try:
                response = requests.post(
                    self.documents_endpoint, json={"documents": batch}, timeout=30
                )

                if response.status_code == 200:
                    total_added += len(batch)
                    print(f"   ✅ Успешно (всего: {total_added})")
                else:
                    print(f"   ❌ Ошибка {response.status_code}")
                    print(f"   Ответ: {response.text[:200]}")
                    break

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                break

        return total_added

    def run_interactive(self):
        """Интерактивный режим загрузки"""
        print("=" * 60)
        print("🚀 УНИВЕРСАЛЬНЫЙ ЗАГРУЗЧИК ДАТАСЕТОВ В RAG")
        print("=" * 60)

        # Проверка сервера
        if not self.check_server():
            print("❌ RAG сервер не запущен!")
            print("   Запустите: python src/api/server.py")
            return

        print("✅ Сервер работает")

        # Выбор типа датасета
        print("\n📁 ВЫБЕРИТЕ ТИП ДАТАСЕТА:")
        print("1. SQuAD (JSON)")
        print("2. Текстовый файл (.txt)")
        print("3. JSON файл")
        print("4. CSV файл")
        print("5. PDF файл")
        print("6. Все файлы из папки")
        print("0. Выход")

        try:
            choice = int(input("Ваш выбор: ").strip())
        except:
            print("🚫 Неверный выбор")
            return

        if choice == 0:
            return

        # Запрос пути
        if choice in [1, 2, 3, 4, 5]:
            filepath = input("Введите путь к файлу: ").strip()
            if not os.path.exists(filepath):
                print(f"❌ Файл не найден: {filepath}")
                return
        elif choice == 6:
            folder_path = input("Введите путь к папке: ").strip()
            if not os.path.exists(folder_path):
                print(f"❌ Папка не найдена: {folder_path}")
                return
            file_ext = input("Расширение файлов (например .txt): ").strip()

        # Запрос количества
        try:
            max_docs = int(input("Максимальное количество документов (0 = все): ").strip())
            if max_docs <= 0:
                max_docs = None
        except:
            max_docs = None

        # Загрузка
        documents = []

        if choice == 1:
            documents = self.load_squad(filepath, max_docs)
        elif choice == 2:
            documents = self.load_txt(filepath, max_docs)
        elif choice == 3:
            text_field = input("Поле с текстом (по умолчанию 'text'): ").strip() or "text"
            documents = self.load_json(filepath, text_field, max_docs)
        elif choice == 4:
            text_column = input("Колонка с текстом (по умолчанию 'text'): ").strip() or "text"
            documents = self.load_csv(filepath, text_column, max_docs)
        elif choice == 5:
            documents = self.load_pdf(filepath, max_docs)
        elif choice == 6:
            documents = self.load_from_folder(folder_path, file_ext, max_docs)

        if not documents:
            print("❌ Не удалось загрузить документы")
            return

        print(f"\n📚 Загружено {len(documents)} документов")

        # Показать примеры
        if input("Показать примеры? (y/n): ").lower() == "y":
            for i, doc in enumerate(documents[:3]):
                print(f"\nПример {i+1}:")
                print(doc[:200] + "..." if len(doc) > 200 else doc)

        # Подтверждение загрузки
        if input(f"\nДобавить {len(documents)} документов в RAG? (y/n): ").lower() == "y":
            added = self.add_to_rag(documents)
            print(f"\n🎯 Добавлено {added} документов")

            # Финальная информация
            try:
                response = requests.get(self.info_endpoint)
                if response.status_code == 200:
                    info = response.json()
                    print(f"📊 Всего в системе: {info.get('total_documents', 0)} документов")
            except:
                pass


def main():
    """Точка входа"""
    loader = DatasetToRAG()
    loader.run_interactive()


if __name__ == "__main__":
    main()
