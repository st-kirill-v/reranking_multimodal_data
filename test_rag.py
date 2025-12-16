"""
Тестовый скрипт для проверки работы RAG системы.
Проверяет загрузку документов, поиск и генерацию ответов.
"""

import sys
import os
import traceback

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.rag import ModularRAG


def test_rag_system():
    """Основной тест RAG системы."""
    print("=" * 60)
    print("ТЕСТ RAG СИСТЕМЫ")
    print("=" * 60)

    # 1. Инициализация
    print("\n1. ИНИЦИАЛИЗАЦИЯ RAG...")
    try:
        rag = ModularRAG()
        print("✓ RAG система создана успешно")
    except Exception as e:
        print(f"✗ Ошибка при создании RAG: {e}")
        return

    # 2. Проверка состояния системы
    print("\n2. СОСТОЯНИЕ СИСТЕМЫ:")
    try:
        status = rag.get_system_status()
        print(f"   Всего документов: {status['total_documents']}")
        print(f"   Индекс построен: {status['index_built']}")
        print(f"   LLM загружена: {status['llm_loaded']}")
        print(f"   Активные модули: {', '.join(status['active_modules'])}")

        for name, module_info in status["modules"].items():
            print(f"\n   Модуль '{name}':")
            print(f"     - Тип: {module_info['type']}")
            print(f"     - Активен: {module_info['active']}")
            print(f"     - Документов: {module_info['doc_count']}")
            print(f"     - Индекс готов: {module_info['index_ready']}")
    except Exception as e:
        print(f"✗ Ошибка при проверке состояния: {e}")

    # 3. Добавление тестовых документов
    print("\n3. ДОБАВЛЕНИЕ ДОКУМЕНТОВ...")
    test_documents = [
        "Машинное обучение — это область искусственного интеллекта, изучающая методы построения алгоритмов, способных обучаться на данных. Основные типы: обучение с учителем, без учителя и с подкреплением.",
        "Нейронные сети — это математические модели, работающие по принципу биологических нейронных сетей. Они состоят из слоёв нейронов и используются для распознавания образов, классификации и прогнозирования.",
        "Трансформеры — это архитектура нейронных сетей для обработки последовательностей, использующая механизм внимания. Они стали основой для моделей BERT, GPT и T5.",
        "BERT — это двунаправленная модель трансформера для предварительного обучения на текстовых данных. Разработана Google в 2018 году.",
        "GPT-3 — это большая языковая модель с 175 миллиардами параметров, способная генерировать человеческий текст. Создана компанией OpenAI.",
        "Глубокое обучение — это подраздел машинного обучения, использующий многослойные нейронные сети. Позволяет решать сложные задачи: распознавание речи, компьютерное зрение.",
        "Векторное представление слов — это метод преобразования слов в числовые векторы, сохраняющие семантические связи между словами.",
        "Ранжирование в поиске — это процесс упорядочивания документов по релевантности запросу пользователя.",
        "RAG (Retrieval-Augmented Generation) — это архитектура, комбинирующая поиск информации с генерацией текста.",
        "EMBEDDING — это техника преобразования текста в числовые векторы для машинного обучения.",
    ]

    try:
        add_result = rag.add_documents(test_documents)
        print(f"✓ Добавлено {len(test_documents)} документов")
        print(f"   Результат: {add_result.get('status', 'unknown')}")
    except Exception as e:
        print(f"✗ Ошибка при добавлении документов: {e}")
        return

    # 4. Построение индекса
    print("\n4. ПОСТРОЕНИЕ ИНДЕКСА...")
    try:
        index_result = rag.build_index()
        print(f"✓ Индекс построен")
        if "details" in index_result:
            for module_name, result in index_result["details"].get("results", {}).items():
                status = result.get("status", "unknown")
                method = result.get("method", "")
                print(f"   {module_name}: {status} ({method})")
    except Exception as e:
        print(f"✗ Ошибка при построении индекса: {e}")

    # 5. Проверка поиска
    print("\n5. ТЕСТ ПОИСКА...")
    test_queries = [
        "Что такое машинное обучение?",
        "Объясни архитектуру трансформеров",
        "Какие бывают нейронные сети?",
        "Что такое BERT?",
    ]

    for query in test_queries[:2]:  # Тестируем только 2 запроса для скорости
        print(f"\n   Запрос: '{query}'")
        try:
            search_result = rag.search(query, n_results=3)

            if "results" in search_result and search_result["results"]:
                print(f"   ✓ Найдено результатов: {len(search_result['results'])}")
                for i, doc in enumerate(search_result["results"][:3], 1):
                    content = doc.get("content", "")
                    preview = content[:100] + "..." if len(content) > 100 else content
                    score = doc.get("score", 0)
                    print(f"     {i}. [{score:.3f}] {preview}")
            else:
                print(f"   ✗ Нет результатов")

        except Exception as e:
            print(f"   ✗ Ошибка поиска: {e}")

    # 6. Тест генерации ответов
    print("\n6. ТЕСТ ГЕНЕРАЦИИ ОТВЕТОВ...")
    test_questions = [
        "Что такое машинное обучение и какие типы бывают?",
        "Объясни что такое трансформеры в NLP",
    ]

    for question in test_questions:
        print(f"\n   Вопрос: '{question}'")
        try:
            answer_result = rag.generate_answer(question, top_k=3)

            if "answer" in answer_result:
                print(f"   ✓ Ответ сгенерирован:")
                print(f"     {answer_result['answer']}")

                if answer_result.get("sources"):
                    print(f"     Использовано источников: {len(answer_result['sources'])}")
                    for i, source in enumerate(answer_result["sources"][:2], 1):
                        print(f"     {i}. {source.get('preview', '')}")
            else:
                print(f"   ✗ Не удалось сгенерировать ответ")

        except Exception as e:
            print(f"   ✗ Ошибка генерации: {e}")

    # 7. Проверка метрик
    print("\n7. ПРОВЕРКА МЕТРИК...")
    try:
        metrics = rag.get_metrics_summary()
        print(f"✓ Метрики собраны")
        if "latency_ms" in metrics:
            lat = metrics["latency_ms"]
            print(f"   Латентность: средняя {lat.get('mean', 0):.1f}ms")

        query_count = metrics.get("query_count", 0)
        print(f"   Всего запросов: {query_count}")
    except Exception as e:
        print(f"✗ Ошибка при получении метрик: {e}")

    # 8. Итоговая информация
    print("\n8. ИТОГОВАЯ ИНФОРМАЦИЯ:")
    try:
        final_status = rag.get_system_status()
        print(f"   Всего документов в системе: {final_status['total_documents']}")

        # Информация о LLM
        llm_info = rag.llm_generator.get_info()
        print(f"   LLM генератор: {llm_info.get('name', 'unknown')}")
        print(f"   Модель загружена: {llm_info.get('model_loaded', False)}")
        print(f"   Устройство: {llm_info.get('device', 'unknown')}")

    except Exception as e:
        print(f"✗ Ошибка при получении итоговой информации: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)


def test_llm_generator():
    """Тест LLM генератора отдельно."""
    print("\n" + "=" * 60)
    print("ТЕСТ LLM ГЕНЕРАТОРА")
    print("=" * 60)

    try:
        from src.core.generators.llm_generator import create_llm_generator

        print("\n1. Создание генератора...")
        generator = create_llm_generator("dialogpt")

        print("✓ Генератор создан")
        info = generator.get_info()
        print(f"   Тип: {info.get('type')}")
        print(f"   Имя: {info.get('name')}")
        print(f"   Загружена: {info.get('model_loaded')}")
        print(f"   Устройство: {info.get('device')}")

        # Тест переписывания текста
        print("\n2. Тест переписывания текста...")
        test_text = "Машинное обучение это очень интересная тема для изучения."

        for style in ["improve", "formal", "simplify"]:
            result = generator.rewrite_text(test_text, style)
            print(f"   Стиль '{style}': {result}")

        # Тест генерации ответа
        print("\n3. Тест генерации ответа...")
        test_context = [
            {
                "content": "Машинное обучение — это область искусственного интеллекта, которая изучает алгоритмы, способные обучаться на данных. Основные подходы: обучение с учителем, без учителя и с подкреплением."
            },
            {
                "content": "Нейронные сети моделируют работу человеческого мозга и состоят из слоёв нейронов. Они используются для распознавания образов, классификации и прогнозирования."
            },
        ]

        question = "Что такое машинное обучение и как оно связано с нейронными сетями?"
        answer = generator.generate_answer(question, test_context)
        print(f"   Вопрос: {question}")
        print(f"   Ответ: {answer}")

    except Exception as e:
        print(f"✗ Ошибка теста LLM: {e}")
        import traceback

        traceback.print_exc()


def test_single_module(module_name: str = "bm25"):
    """Тест отдельного модуля поиска."""
    print(f"\n" + "=" * 60)
    print(f"ТЕСТ МОДУЛЯ {module_name.upper()}")
    print("=" * 60)

    from src.core.module_manager import ModuleManager

    manager = ModuleManager()

    # Создаем и регистрируем модуль
    if module_name == "bm25":
        from src.core.modules.bm25_module import BM25Module

        module = BM25Module(name="test_bm25", language="russian")
    elif module_name == "e5":
        from src.core.modules.e5_module import E5Module

        module = E5Module(name="test_e5", device="cpu")
    else:
        print(f"✗ Неизвестный модуль: {module_name}")
        return

    manager.register_search_module(module, activate=True)

    # Добавляем документы
    test_docs = [
        "Машинное обучение изучает алгоритмы, которые учатся на данных.",
        "Нейронные сети используются для распознавания образов.",
        "Трансформеры — это архитектура для обработки текста.",
    ]

    print(f"\n1. Добавление документов в {module_name}...")
    result = manager.add_documents(test_docs)
    print(f"   Результат: {result}")

    print(f"\n2. Поиск в {module_name}...")
    query = "машинное обучение"
    search_results = manager.search(query, n_results=2, strategy="simple")

    if search_results.get("results"):
        print(f"   Запрос: '{query}'")
        for i, doc in enumerate(search_results["results"][:2], 1):
            content = doc.get("content", "")
            preview = content[:80] + "..." if len(content) > 80 else content
            score = doc.get("score", 0)
            print(f"   {i}. [{score:.3f}] {preview}")
    else:
        print(f"   ✗ Нет результатов для '{query}'")


if __name__ == "__main__":
    print("Запуск тестов RAG системы...")

    # Запускаем тесты
    test_rag_system()
