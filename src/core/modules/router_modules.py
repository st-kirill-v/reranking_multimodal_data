"""
Умный роутер для комплексных запросов (график + таблица + код + текст)
Заменит SimpleRouter
"""

from typing import List, Dict, Optional
import re


class SmartRouter:
    """
    🎯 УМНЫЙ РОУТЕР: понимает комплексные запросы

    Примеры:
    - "график продаж" → ["bm25", "clip"]
    - "таблица данных" → ["bm25", "layoutlm"]
    - "график и таблица" → ["bm25", "clip", "layoutlm"]
    - "код функции" → ["bm25", "e5"]
    - "просто текст" → ["bm25", "e5"] (по умолчанию)
    """

    def __init__(self):
        # 🔑 Ключевые слова для каждой категории
        self.categories = {
            "visual": {  # Визуальный контент
                "keywords": [
                    "график",
                    "диаграмма",
                    "изображение",
                    "фото",
                    "картинка",
                    "схема",
                    "чертеж",
                    "визуализация",
                    "рисунок",
                    "иллюстрация",
                    "скриншот",
                    "picture",
                    "chart",
                    "diagram",
                    "image",
                    "photo",
                ],
                "modules": ["clip"],
            },
            "table": {  # Таблицы и структура
                "keywords": [
                    "таблица",
                    "табличный",
                    "столбец",
                    "строка",
                    "ячейка",
                    "excel",
                    "csv",
                    "табличные данные",
                    "сводная",
                    "матрица",
                    "табличка",
                    "таблиц",
                    "table",
                    "spreadsheet",
                    "column",
                    "row",
                ],
                "modules": ["layoutlm"],
            },
            "code": {  # Программный код
                "keywords": [
                    "код",
                    "программа",
                    "функция",
                    "алгоритм",
                    "скрипт",
                    "исходник",
                    "синтаксис",
                    "библиотека",
                    "модуль",
                    "класс",
                    "метод",
                    "переменная",
                    "code",
                    "program",
                    "function",
                    "algorithm",
                ],
                "modules": ["e5"],  # E5 хорошо для кода
            },
        }

    def route(self, query: str, document_type: Optional[str] = None) -> List[str]:
        """
        🎯 ОСНОВНОЙ МЕТОД: определяет какие модули использовать

        Логика:
        1. BM25 всегда (быстрый лексический поиск)
        2. Ищем ключевые слова ВО ВСЕХ категориях
        3. Добавляем модули из ВСЕХ совпавших категорий
        4. Если ничего не найдено → E5 по умолчанию
        """
        query_lower = query.lower()
        selected_modules = ["bm25"]  # 🎯 BM25 ВСЕГДА

        # 🔍 Проверяем ВСЕ категории (не только первую!)
        for category_name, category in self.categories.items():
            for keyword in category["keywords"]:
                if keyword in query_lower:
                    # Добавляем ВСЕ модули из этой категории
                    for module in category["modules"]:
                        if module not in selected_modules:
                            selected_modules.append(module)
                    break  # Достаточно одного ключевого слова в категории

        # 🎯 Если только BM25 → добавляем E5 по умолчанию
        if len(selected_modules) == 1:
            selected_modules.append("e5")

        return selected_modules

    def explain(self, query: str, document_type: Optional[str] = None) -> Dict:
        """
        🔍 Объясняет почему выбраны те или иные модули
        Полезно для отладки
        """
        query_lower = query.lower()
        explanation = {
            "query": query,
            "selected_modules": self.route(query, document_type),
            "matched_keywords": [],
            "matched_categories": [],
        }

        # Анализируем какие категории сработали
        for category_name, category in self.categories.items():
            matched_keywords = []
            for keyword in category["keywords"]:
                if keyword in query_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                explanation["matched_categories"].append(
                    {
                        "name": category_name,
                        "keywords": matched_keywords,
                        "modules": category["modules"],
                    }
                )
                explanation["matched_keywords"].extend(matched_keywords)

        return explanation


# Версия с логированием для разработки
class DebugRouter(SmartRouter):
    def route(self, query: str, document_type=None):
        # Просто вызываем родительский метод без explain в route
        result = super().route(query, document_type)

        # Логирование отдельно
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG ROUTER")
        print(f"{'='*60}")
        print(f"📋 Запрос: {query}")
        print(f"🎯 Результат: {result}")

        # Вместо self.explain() делаем упрощенную версию
        query_lower = query.lower()
        matched_keywords = []
        for category_name, category in self.categories.items():
            for keyword in category["keywords"]:
                if keyword in query_lower:
                    matched_keywords.append(keyword)

        if matched_keywords:
            print(f"🔑 Найдены ключевые слова: {matched_keywords}")

        print(f"{'='*60}\n")

        return result
