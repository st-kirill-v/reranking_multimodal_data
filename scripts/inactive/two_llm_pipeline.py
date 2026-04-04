import sys
from pathlib import Path
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.full_pipeline import full_search, expand_with_neighbors
from src.core.generators.qwen_vl_generator import create_qwen_generator
from src.core.generators.yandex_gpt_generator import create_llm_generator


class TwoLLMPipeline:
    """Двухэтапный RAG пайплайн: Qwen2-VL извлекает контекст, YandexGPT генерирует ответ"""

    def __init__(self, device: str = "cuda"):
        self.device = device

        print("Загрузка LLM-1: Qwen2-VL-7B (извлечение контекста)...")

        self.extractor = create_qwen_generator(device=device)
        print("Контекстный экстрактор загружен")

        print("\nЗагрузка LLM-2: YandexGPT (генерация ответа)...")

        self.answer_generator = create_llm_generator(generator_type="yandexgpt")
        print("Генератор ответов загружен")

        print("\nОбе модели готовы!")

    def get_search_results(self, query: str) -> list:
        """Запускает поиск и возвращает 15 страниц PNG"""
        print(f"\nПоиск по запросу: {query[:80]}...")

        candidates, _ = full_search(query, top_k_initial=400, top_k_rerank=150, final_k=30)
        candidates_with_neighbors = expand_with_neighbors(candidates, max_pages=15)

        images = []
        for cand in candidates_with_neighbors[:15]:
            try:
                img_path = Path(cand["path"])
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
            except Exception as e:
                continue

        print(f"  Загружено {len(images)} страниц для анализа")
        return images

    def extract_context(self, query: str, context_images: list) -> str:
        """LLM-1: Извлекает релевантный контекст из 15 страниц"""

        prompt = f"""Here are {len(context_images)} document pages.

Find information that answers this question: "{query}"

Extract ALL relevant text, numbers, and table data. Be thorough.

Output ONLY the extracted information, nothing else. Do not explain what you are doing.

Extracted information:"""

        answer = self.extractor.generate_answer(prompt, context_images)
        return answer.strip()

    def generate_answer(self, context: str, question: str) -> str:
        """LLM-2: Генерирует краткий ответ по контексту"""
        answer = self.answer_generator.generate_answer(question, [{"content": context}])
        return answer.strip()

    def run(self, query: str, verbose: bool = True) -> str:
        """Запускает полный пайплайн"""
        if verbose:
            print(f"\nВопрос: {query}")

        # Шаг 1: Поиск
        images = self.get_search_results(query)

        if not images:
            return "NOT FOUND - No relevant pages"

        # Шаг 2: Извлечение контекста (Qwen)
        if verbose:
            print("\nШаг 1: Извлечение релевантного контекста с помощью Qwen2-VL...")

        context = self.extract_context(query, images)

        if verbose:
            print(f"   Длина контекста: {len(context)} символов")
            print(f"   Превью: {context[:300]}...")

        # Шаг 3: Генерация ответа (YandexGPT)
        if verbose:
            print("\nШаг 2: Генерация краткого ответа с помощью YandexGPT...")

        answer = self.generate_answer(context, query)

        if verbose:
            print(f"\nФинальный ответ: {answer}")

        return answer


if __name__ == "__main__":
    print("Двухэтапный RAG пайплайн")
    print("   Qwen2-VL (извлечение контекста) → YandexGPT (генерация ответа)")

    pipeline = TwoLLMPipeline(device="cuda")

    test_query = "What is the aligned BLEU score for GL → EN?"

    answer = pipeline.run(test_query, verbose=True)

    print("\nПайплайн завершён")
