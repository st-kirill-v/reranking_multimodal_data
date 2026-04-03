import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv


class YandexGPTRAGGenerator:

    def __init__(self, folder_id: str = None, api_key: str = None):
        load_dotenv()

        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._model = "yandexgpt"

        self._validate_credentials()

    def _validate_credentials(self):
        missing = []
        if not self.folder_id:
            missing.append("YANDEX_FOLDER_ID")
        if not self.api_key:
            missing.append("YANDEX_API_KEY")
        if missing:
            raise ValueError(f"Missing credentials: {', '.join(missing)}")

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        if not query or not query.strip():
            return "Please provide a question."

        if not context_docs:
            return f"No relevant information found for: '{query}'"

        contexts = []
        for doc in context_docs[:5]:
            content = doc.get("content", "")
            if content and len(content) > 10:
                contexts.append(content[:4000])

        if not contexts:
            return f"Could not extract information for: '{query}'"

        context_text = "\n\n".join([f"[Document {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)])

        # ============================================
        # ГИБКИЙ ПРОМПТ ДЛЯ РАЗНЫХ ТИПОВ ОТВЕТОВ
        # ============================================
        system_prompt = """You are a precise RAG assistant. Answer based ONLY on the provided context.

RULES:
1. If the question asks for a NUMBER → output only the number (e.g., "65", "67.5")
2. If the question asks for a NAME/TERM → output only the term (e.g., "KGLM", "softmax")
3. If the question asks for a DESCRIPTION → output a complete but concise sentence
4. If the answer is NOT in the context → say "NOT FOUND"
5. Be concise but complete. Match the expected answer format.

Examples:
- Q: "What is the accuracy?" → A: "65"
- Q: "Which model?" → A: "KGLM"
- Q: "What is the main challenge?" → A: "The main challenge is incorporating factual knowledge."

Now answer the question based ONLY on the context:"""

        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json",
        }

        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 500},
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
                return answer.strip('"').strip("'").strip()

            elif response.status_code == 429:
                return "Rate limit exceeded"
            else:
                return f"API error ({response.status_code})"

        except requests.exceptions.Timeout:
            return "Request timeout"
        except requests.exceptions.ConnectionError:
            return "Connection error"
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "llm_generator",
            "name": "yandexgpt",
            "model": self._model,
            "temperature": 0.4,
            "max_tokens": 500,
            "api": "Yandex GPT",
        }


def create_llm_generator(generator_type: str = "yandexgpt", **kwargs):
    if generator_type.lower() in ["yandexgpt", "yandex", "gpt"]:
        return YandexGPTRAGGenerator(
            folder_id=kwargs.get("folder_id"), api_key=kwargs.get("api_key")
        )
    else:
        raise ValueError(f"Generator type '{generator_type}' not supported")
