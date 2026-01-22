import requests
from typing import List, Dict, Any


class YandexGPTRAGGenerator:
    """Генератор ответов на основе Яндекс GPT для RAG системы"""

    def __init__(
        self,
        folder_id: str = "b1gurdq8cah90hv2cpfq",
        api_key: str = "AQVN3c4ppB88MuPpr3Cevy3XANS_DsVuUcbsWHeJ",
    ):
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.folder_id = folder_id
        self.api_key = api_key
        self._model = "yandexgpt"

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

                # Очистка ответа (как в твоём старом коде)
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
            "folder_id": self.folder_id[:10] + "...",  # Частично для безопасности
        }


def create_llm_generator(generator_type: str = "yandexgpt", **kwargs):
    """Фабричная функция для совместимости с твоим старым кодом"""
    if generator_type.lower() in ["yandexgpt", "yandex", "gpt"]:
        return YandexGPTRAGGenerator()
    else:
        # Fallback на другие генераторы если нужно
        raise ValueError(f"Тип генератора '{generator_type}' не поддерживается")


# Пример использования
if __name__ == "__main__":
    generator = create_llm_generator("yandexgpt")

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
