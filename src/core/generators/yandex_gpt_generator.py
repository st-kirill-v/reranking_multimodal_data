import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv


class YandexGPTRAGGenerator:
    """Генератор ответов на основе Яндекс GPT для RAG системы"""

    def __init__(
        self,
        folder_id: str = None,
        api_key: str = None,
    ):
        """
        Args:
            folder_id: ID каталога Yandex Cloud (берётся из .env или параметра)
            api_key: API ключ Yandex Cloud (берётся из .env или параметра)
        """
        load_dotenv()  # Загружаем переменные окружения из .env файла

        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._model = "yandexgpt"

        self._validate_credentials()

    def _validate_credentials(self):
        """Проверяет наличие необходимых учётных данных"""
        missing = []

        if not self.folder_id:
            missing.append("YANDEX_FOLDER_ID")
        if not self.api_key:
            missing.append("YANDEX_API_KEY")

        if missing:
            error_msg = (
                f"Не найдены обязательные учётные данные: {', '.join(missing)}.\n"
                "Добавьте их в файл .env или передайте в конструктор класса.\n"
                "Пример .env файла:\n"
                "YANDEX_FOLDER_ID=your_folder_id_here\n"
                "YANDEX_API_KEY=your_api_key_here"
            )
            raise ValueError(error_msg)

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Генерирует ответ на основе контекста

        Args:
            query: Вопрос пользователя
            context_docs: Список документов с полем 'content'

        Returns:
            Ответ модели
        """
        if not query or not query.strip():
            return "Пожалуйста, задайте вопрос."

        if not context_docs:
            return f"Не найдено релевантной информации для запроса: '{query}'"

        # Формируем контекст из документов (первые 3, каждый до 300 символов)
        contexts = []
        for doc in context_docs[:3]:
            content = doc.get("content", "")
            if content and len(content) > 10:
                contexts.append(content[:300])

        if not contexts:
            return f"Не удалось найти информацию по запросу: '{query}'"

        # Объединяем контекст
        context_text = "\n".join([f"[Документ {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])

        # Системный промпт для RAG
        system_prompt = """Ты — RAG ассистент. Отвечай строго на основе предоставленного контекста.
        Задача: предоставлять точные, информативные ответы на основе документов.
        Цель: помочь пользователю найти нужную информацию без выхода за рамки контекста.
        Если в контексте нет ответа — честно скажи об этом."""

        # Пользовательский промпт
        user_prompt = f"""Информация для ответа:
{context_text}

Вопрос: {query}

Ответ ассистента (только на основе контекста):"""

        # Заголовки запроса
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json",
        }

        # Тело запроса
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.7, "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                answer = result["result"]["alternatives"][0]["message"]["text"]

                # Очистка ответа
                answer = answer.strip('"').strip("'").strip()

                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
                if answer and answer[-1] not in ".!?;":
                    answer = answer + "."

                return answer

            elif response.status_code == 429:
                return "Слишком много запросов. Попробуйте позже."
            else:
                return f"Ошибка API ({response.status_code}): {response.text[:200]}"

        except requests.exceptions.Timeout:
            return "Таймаут при подключении к серверу"
        except requests.exceptions.ConnectionError:
            return "Ошибка подключения к серверу"
        except Exception as e:
            return f"Неизвестная ошибка: {str(e)[:200]}"

    def rewrite_text(self, text: str, style: str = "improve") -> str:
        """Переписывает текст в заданном стиле (опционально)"""
        # Можно добавить позже если нужно
        return text

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о генераторе (для совместимости с интерфейсом)"""
        return {
            "type": "llm_generator",
            "name": "yandexgpt",
            "model": self._model,
            "model_loaded": True,
            "temperature": 0.7,
            "max_tokens": 2000,
            "api": "Yandex GPT",
            "folder_id": (
                self.folder_id[:4] + "..." + self.folder_id[-4:] if self.folder_id else "not_set"
            ),
        }


def create_llm_generator(generator_type: str = "yandexgpt", **kwargs):
    """Фабричная функция для совместимости с твоим старым кодом"""
    if generator_type.lower() in ["yandexgpt", "yandex", "gpt"]:
        return YandexGPTRAGGenerator(
            folder_id=kwargs.get("folder_id"), api_key=kwargs.get("api_key")
        )
    else:
        # Fallback на другие генераторы если нужно
        raise ValueError(f"Тип генератора '{generator_type}' не поддерживается")


# Пример использования
if __name__ == "__main__":
    # Пример 1: Использование с переменными окружения
    generator = create_llm_generator("yandexgpt")

    # Пример 2: Использование с прямым указанием параметров
    # generator = create_llm_generator(
    #     "yandexgpt",
    #     folder_id="your_folder_id_here",
    #     api_key="your_api_key_here"
    # )

    context_docs = [
        {
            "content": "Яндекс GPT — это языковая модель от компании Яндекс. Она умеет генерировать тексты, отвечать на вопросы и выполнять другие задачи."
        },
        {
            "content": "Бесплатный тариф Яндекс GPT предоставляет 4000 токенов в месяц. Токен — это часть слова, используемая для обработки текста."
        },
    ]

    answer = generator.generate_answer(
        query="Сколько токенов в бесплатном тарифе Яндекс GPT?", context_docs=context_docs
    )

    print(f"Ответ: {answer}")
