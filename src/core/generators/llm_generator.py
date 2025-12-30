"""
Генератор на основе локальной модели DialoGPT-small.
Используется для переписывания текста и генерации ответов в RAG системе.
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class DialoGPTGenerator:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Несколько возможных путей для поиска модели
            possible_paths = [
                r"d:\project\reranking_multimodal_data\models\dialogpt-small",
                r"./models/dialogpt-small",
                r"models/dialogpt-small",
                "microsoft/DialoGPT-small",  # Для автоматической загрузки
            ]

            # Ищем существующий путь
            for path in possible_paths:
                if os.path.exists(path) or "microsoft/" in path:
                    self.model_path = path
                    print(f"Выбран путь модели: {path}")
                    break
            else:
                self.model_path = possible_paths[0]  # Используем первый путь по умолчанию
        else:
            self.model_path = model_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство для модели: {self.device}")

        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def _verify_model_files(self) -> tuple[bool, str]:
        """Проверяет наличие файлов модели."""
        # Если это хаб-модель, не проверяем локальные файлы
        if (
            "microsoft/" in self.model_path
            or "/" in self.model_path
            and not os.path.exists(self.model_path)
        ):
            return True, "используется модель из Hugging Face Hub"

        if not os.path.exists(self.model_path):
            return False, f"папка не существует: {self.model_path}"

        required_files = ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
        model_files = ["model.safetensors", "pytorch_model.bin"]

        existing_files = os.listdir(self.model_path)

        missing_required = [f for f in required_files if f not in existing_files]
        has_model = any(mf in existing_files for mf in model_files)

        if missing_required:
            return False, f"отсутствуют файлы: {missing_required}"

        if not has_model:
            return False, "не найден файл модели"

        return True, "все файлы на месте"

    def load(self, force_reload: bool = False):
        """Загружает модель и токенизатор."""
        if self.is_loaded and not force_reload:
            print("Модель уже загружена")
            return

        print(f"Загрузка модели из: {self.model_path}")

        # Проверяем файлы только для локальных моделей
        if not "microsoft/" in self.model_path:
            is_valid, message = self._verify_model_files()
            if not is_valid:
                print(f"Предупреждение: {message}")
                print("Попытка загрузки из Hugging Face Hub...")
                self.model_path = "microsoft/DialoGPT-small"

        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # Перемещаем на устройство
            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            print(f"Модель успешно загружена на устройство: {self.device}")
            print(f"Размер словаря: {len(self.tokenizer)}")

        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели: {e}")

    def rewrite_text(self, text: str, style: str = "improve") -> str:
        """Переписывает текст в заданном стиле."""
        if not text or not text.strip():
            return text

        if not self.is_loaded:
            self.load()

        style_instructions = {
            "improve": "улучши и перепиши этот текст:",
            "simplify": "упрости этот текст для понимания:",
            "formal": "сделай текст более формальным и профессиональным:",
            "expand": "расширь текст с дополнительными деталями:",
            "summary": "создай краткое содержание текста:",
        }

        instruction = style_instructions.get(style, style_instructions["improve"])

        # Обрезаем текст если слишком длинный
        if len(text) > 500:
            text = text[:497] + "..."

        # Формируем промпт
        prompt = f"{instruction}\n{text}\nРезультат:"

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Извлекаем результат
            if "Результат:" in full_response:
                result = full_response.split("Результат:")[-1].strip()
            elif "result:" in full_response:
                result = full_response.split("result:")[-1].strip()
            else:
                # Пытаемся извлечь текст после промпта
                result = full_response.replace(prompt, "").strip()

            # Очищаем результат
            result = result.split("\n")[0].strip()
            if not result or len(result) < 3:
                return text

            return result

        except Exception as e:
            print(f"Ошибка при переписывании текста: {e}")
            return text

    def generate_answer(self, query: str, context_docs: list) -> str:
        """Генерирует ответ на основе контекста."""
        if not query or not query.strip():
            return "Пожалуйста, задайте вопрос."

        if not self.is_loaded:
            self.load()

        if not context_docs:
            return f"Не найдено релевантной информации для запроса: '{query}'"

        # Формируем контекст из документов
        contexts = []
        for doc in context_docs[:3]:  # Берем до 3 документов
            content = doc.get("content", "")
            if content and len(content) > 10:
                contexts.append(content[:300])  # Обрезаем каждый документ

        if not contexts:
            return f"Не удалось найти информацию по запросу: '{query}'"

        context_text = "\n".join([f"[Документ {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""Пользователь: {query}

Информация для ответа:
{context_text}

Ассистент: На основе этой информации, """

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Извлекаем ответ (ищем ответ после "Ассистент: ")
            if "Ассистент:" in full_response:
                answer = full_response.split("Ассистент:")[-1].strip()
            elif "assistant:" in full_response:
                answer = full_response.split("assistant:")[-1].strip()
            else:
                # Если не нашли маркер, берем всё после промпта
                answer = full_response.replace(prompt, "").strip()

            # Если ответ пустой или слишком короткий
            if not answer or len(answer) < 3:
                # Fallback: берем первую часть контекста
                if contexts:
                    answer = f"Согласно информации: {contexts[0][:100]}..."
                else:
                    answer = "Не могу сформулировать ответ на основе доступной информации."

            # Удаляем лишние кавычки и символы
            answer = answer.strip('"').strip("'").strip()

            # Форматируем первое предложение
            if answer and answer[0].islower():
                answer = answer[0].upper() + answer[1:]
            if answer and answer[-1] not in ".!?;":
                answer = answer + "."

            return answer

        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            # Возвращаем fallback ответ
            return f"На основе найденной информации: {contexts[0][:100]}..."

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о генераторе."""
        return {
            "type": "llm_generator",
            "name": "dialogpt",
            "model_loaded": self.is_loaded,
            "device": self.device,
            "model_path": self.model_path,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
        }


class TemplateGenerator:
    """Простой шаблонный генератор для fallback."""

    def __init__(self):
        self.name = "template_generator"
        self.is_loaded = True

    def rewrite_text(self, text: str, style: str = "improve") -> str:
        """Просто возвращает исходный текст."""
        return text

    def generate_answer(self, query: str, context_docs: list) -> str:
        """Генерирует ответ на основе шаблона."""
        if not context_docs:
            return f"Не найдено результатов для запроса: '{query}'"

        answer_parts = [f"Ответ на запрос: '{query}'", ""]

        for i, doc in enumerate(context_docs[:3], 1):
            content = doc.get("content", "")
            if content:
                # Обрезаем длинный контент
                preview = content[:150] + "..." if len(content) > 150 else content
                answer_parts.append(f"{i}. {preview}")

        return "\n".join(answer_parts)

    def get_info(self) -> Dict[str, Any]:
        return {"type": "template_generator", "name": self.name, "model_loaded": True}


def create_llm_generator(generator_type: str = "dialogpt", model_path: Optional[str] = None):
    """
    Создает генератор LLM.

    Args:
        generator_type: Тип генератора ("dialogpt", "dialoGPT", "template")
        model_path: Путь к модели (опционально)

    Returns:
        Экземпляр генератора
    """
    # Нормализуем тип генератора
    generator_type = generator_type.lower().strip()

    # Поддерживаем разные варианты написания
    if generator_type in ["dialogpt", "dialo-gpt", "dialo_gpt", "gpt"]:
        try:
            print(f"Создание DialoGPT генератора (тип: {generator_type})")
            generator = DialoGPTGenerator(model_path)
            generator.load()
            print("DialoGPT генератор успешно создан")
            return generator
        except Exception as e:
            print(f"Ошибка при создании DialoGPT генератора: {e}")
            print("Использую шаблонный генератор как fallback")
            return TemplateGenerator()

    elif generator_type == "template":
        print("Создание шаблонного генератора")
        return TemplateGenerator()

    else:
        raise ValueError(
            f"Неизвестный тип генератора: {generator_type}. "
            f"Доступные типы: 'dialogpt', 'template'"
        )
