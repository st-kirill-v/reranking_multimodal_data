"""
умный роутер для комплексных запросов с поддержкой визуального контента, таблиц и кода.
"""

from typing import List, Dict, Optional
import re


class SmartRouter:
    def __init__(self):
        self.categories = {
            "visual": {
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
            "table": {
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
            "code": {
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
                "modules": ["e5"],
            },
        }

    def route(self, query: str, document_type: Optional[str] = None) -> List[str]:
        query_lower = query.lower()
        selected_modules = ["bm25"]

        for category_name, category in self.categories.items():
            for keyword in category["keywords"]:
                if keyword in query_lower:
                    for module in category["modules"]:
                        if module not in selected_modules:
                            selected_modules.append(module)
                    break

        if len(selected_modules) == 1:
            selected_modules.append("e5")

        return selected_modules

    def explain(self, query: str, document_type: Optional[str] = None) -> Dict:
        query_lower = query.lower()
        explanation = {
            "query": query,
            "selected_modules": self.route(query, document_type),
            "matched_keywords": [],
            "matched_categories": [],
        }

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


class DebugRouter(SmartRouter):
    def route(self, query: str, document_type=None):
        result = super().route(query, document_type)

        query_lower = query.lower()
        matched_keywords = []
        for category_name, category in self.categories.items():
            for keyword in category["keywords"]:
                if keyword in query_lower:
                    matched_keywords.append(keyword)

        return result
