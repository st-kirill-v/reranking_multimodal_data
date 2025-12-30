"""
Тестовый скрипт для проверки работы RAG системы с каскадным поиском BM25+E5
"""

import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.rag import ModularRAG


def test_rag_system():
    """Основной тест RAG системы с каскадным поиском BM25+E5"""
    print("=" * 60)
    print("ТЕСТ RAG СИСТЕМЫ С КАСКАДНЫМ ПОИСКОМ (BM25 + E5)")
    print("=" * 60)

    # 1. Инициализация
    print("\n1. ИНИЦИАЛИЗАЦИЯ RAG...")
    try:
        rag = ModularRAG()

        if "e5_reranker" not in rag.manager.active_searchers:
            rag.manager.active_searchers.append("e5_reranker")
            print("Добавил e5_reranker в активные модули поиска")

        print("RAG система создана успешно")
    except Exception as e:
        print(f"Ошибка при создании RAG: {e}")
        traceback.print_exc()
        return

    # 2. Проверка состояния системы
    print("\n2. СОСТОЯНИЕ СИСТЕМЫ:")
    try:
        status = rag.get_system_status()
        print(f"Всего документов: {status['total_documents']}")
        print(f"Индекс построен: {status['index_built']}")
        print(f"LLM загружена: {status['llm_loaded']}")
        print(f"Активные модули: {', '.join(status['active_modules'])}")

        for name, module_info in status["modules"].items():
            print(f"\nМодуль '{name}':")
            print(f"  Тип: {module_info['type']}")
            print(f"  Активен: {module_info['active']}")
            print(f"  Документов: {module_info['doc_count']}")
            print(f"  Индекс готов: {module_info['index_ready']}")
    except Exception as e:
        print(f"Ошибка при проверке состояния: {e}")

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
        print(f"Добавлено {len(test_documents)} документов")
        print(f"Результат: {add_result.get('status', 'unknown')}")

        if "details" in add_result:
            for module_name, result in add_result["details"].items():
                if result.get("status") == "success":
                    count = result.get("added", result.get("total_documents", 0))
                    print(f"{module_name}: {count} документов")
    except Exception as e:
        print(f"Ошибка при добавлении документов: {e}")
        traceback.print_exc()
        return

    # 4. Построение индекса
    print("\n4. ПОСТРОЕНИЕ ИНДЕКСА...")
    try:
        index_result = rag.build_index()
        print("Индекс построен")
        if "details" in index_result:
            for module_name, result in index_result["details"].get("results", {}).items():
                status = result.get("status", "unknown")
                method = result.get("method", "")
                details = result.get("details", "")
                print(f"{module_name}: {status} ({method}) {details}")
    except Exception as e:
        print(f"Ошибка при построении индекса: {e}")
        traceback.print_exc()

    # 5. Проверка поиска с каскадным BM25+E5
    print("\n5. ТЕСТ КАСКАДНОГО ПОИСКА (BM25 → E5)...")
    test_queries = [
        ("Что такое машинное обучение?", "Простой запрос"),
        ("Объясни архитектуру трансформеров", "Запрос про трансформеры"),
        ("Что такое BERT и GPT?", "Запрос про модели NLP"),
    ]

    for query, description in test_queries:
        print(f"\nЗапрос: '{query}' ({description})")
        try:
            search_result = rag.search(query, n_results=3, strategy="all")

            if "results" in search_result and search_result["results"]:
                print(f"Найдено результатов: {len(search_result['results'])}")

                modules_used = search_result.get("modules_used", [])
                print(f"Использованы модули: {', '.join(modules_used)}")

                for i, doc in enumerate(search_result["results"][:3], 1):
                    content = doc.get("content", "")
                    preview = content[:80] + "..." if len(content) > 80 else content
                    score = doc.get("score", 0)
                    method = doc.get("method", "unknown")

                    if method == "bm25+e5":
                        e5_score = doc.get("e5_score", 0)
                        bm25_score = doc.get("bm25_score", 0)
                        print(
                            f"{i}. [{score:.3f}] [{method}] E5:{e5_score:.3f}+BM25:{bm25_score:.3f} - {preview}"
                        )
                    else:
                        print(f"{i}. [{score:.3f}] [{method}] - {preview}")
            else:
                print("Нет результатов")

            if search_result.get("all_results"):
                print("Подробно по модулям:")
                for module_name, module_results in search_result["all_results"].items():
                    if module_results:
                        print(f"  {module_name}: {len(module_results)} результатов")

        except Exception as e:
            print(f"Ошибка поиска: {e}")
            traceback.print_exc()

    # 6. Тест генерации ответов (упрощенный)
    print("\n6. ТЕСТ ГЕНЕРАЦИИ ОТВЕТОВ (упрощенный)...")
    test_questions = [
        "Что такое машинное обучение?",
        "Объясни что такое трансформеры",
    ]

    for question in test_questions:
        print(f"\nВопрос: '{question}'")
        try:
            search_result = rag.search(question, n_results=2, strategy="all")

            if search_result.get("results"):
                print(f"Найдено {len(search_result['results'])} источников для ответа")

                print("Лучшие источники:")
                for i, doc in enumerate(search_result["results"][:2], 1):
                    content = doc.get("content", "")
                    score = doc.get("score", 0)
                    method = doc.get("method", "unknown")
                    preview = content[:120] + "..." if len(content) > 120 else content
                    print(f"{i}. [{score:.3f}] [{method}] {preview}")

                    if method == "bm25+e5":
                        e5_score = doc.get("e5_score", 0)
                        bm25_score = doc.get("bm25_score", 0)
                        print(f"  (E5: {e5_score:.3f}, BM25: {bm25_score:.3f})")
            else:
                print("Не найдено источников для ответа")

        except Exception as e:
            print(f"Ошибка: {e}")

    # 7. Проверка метрик
    print("\n7. ПРОВЕРКА МЕТРИК...")
    try:
        metrics = rag.get_metrics_summary()
        print("Метрики собраны")

        query_count = metrics.get("query_count", 0)
        print(f"Всего запросов: {query_count}")

        if "module_stats" in metrics:
            print("Статистика по модулям:")
            for module_name, stats in metrics["module_stats"].items():
                calls = stats.get("call_count", 0)
                avg_time = stats.get("avg_time_ms", 0)
                print(f"  {module_name}: {calls} вызовов, ~{avg_time:.1f}ms")

    except Exception as e:
        print(f"Ошибка при получении метрик: {e}")

    # 8. Итоговая информация
    print("\n8. ИТОГОВАЯ ИНФОРМАЦИЯ:")
    try:
        final_status = rag.get_system_status()
        print(f"Всего документов в системе: {final_status['total_documents']}")
        print(f"Активные модули поиска: {', '.join(final_status['active_modules'])}")

        for name, module_info in final_status["modules"].items():
            if module_info["active"]:
                print(
                    f"Модуль {name} ({module_info['type']}): {module_info['doc_count']} документов, индекс: {'✓' if module_info['index_ready'] else '✗'}"
                )

        llm_info = rag.llm_generator.get_info()
        print(f"LLM генератор: {llm_info.get('name', 'unknown')}")
        print(f"Модель загружена: {'✓' if llm_info.get('model_loaded', False) else '✗'}")
        print(f"Устройство: {llm_info.get('device', 'unknown')}")

    except Exception as e:
        print(f"Ошибка при получении итоговой информации: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)


def test_cascaded_search_only():
    """Тест только каскадного поиска без генерации"""
    print("=" * 60)
    print("ТЕСТ КАСКАДНОГО ПОИСКА (BM25 → E5)")
    print("=" * 60)

    from src.core.module_manager import ModuleManager

    manager = ModuleManager(storage_path="cache/cascade_test")

    bm25 = manager.register_bm25_module(
        name="bm25_cascade", language="russian", k1=1.2, b=0.75, activate=True
    )

    e5 = manager.register_e5_module(
        name="e5_cascade",
        bm25_module_name="bm25_cascade",
        top_k_candidates=50,
        device="cpu",
        activate=True,
    )

    test_docs = [
        "Трансформеры — архитектура нейронных сетей с механизмом внимания.",
        "BERT — модель трансформера от Google для понимания языка.",
        "GPT-3 — большая языковая модель от OpenAI на основе трансформеров.",
        "Машинное обучение изучает алгоритмы которые учатся на данных.",
        "EMBEDDING — техника преобразования текста в векторы.",
    ]

    manager.add_documents(test_docs)
    print(f"Добавлено {len(test_docs)} документов")

    test_cases = [
        ("трансформеры", "Поиск по ключевому слову"),
        ("архитектура трансформеров", "Расширенный запрос"),
        ("BERT GPT модели", "Запрос с несколькими сущностями"),
    ]

    for query, description in test_cases:
        print(f"\nЗапрос: '{query}' ({description})")

        result = manager.search(query, top_k=3, strategy="all")

        modules_used = result.get("modules_used", [])
        print(f"Модули: {modules_used}")

        if result.get("results"):
            for i, doc in enumerate(result["results"][:3], 1):
                score = doc.get("score", 0)
                method = doc.get("method", "unknown")
                preview = doc.get("content", "")[:70] + "..."

                if method == "bm25+e5":
                    e5_score = doc.get("e5_score", 0)
                    bm25_score = doc.get("bm25_score", 0)
                    print(f"{i}. [{score:.3f}] {method} (E5:{e5_score:.3f}+BM25:{bm25_score:.3f})")
                    print(f"   {preview}")
                else:
                    print(f"{i}. [{score:.3f}] {method}")
                    print(f"   {preview}")
        else:
            print("Нет результатов")

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 60)


if __name__ == "__main__":
    print("Запуск тестов RAG системы...")

    print("\nВыберите тест:")
    print("1. Полный тест RAG системы")
    print("2. Только каскадный поиск (BM25 → E5)")
    print("3. Оба теста")

    choice = input("\nВведите номер (1-3): ").strip()

    if choice == "1":
        test_rag_system()
    elif choice == "2":
        test_cascaded_search_only()
    elif choice == "3":
        test_cascaded_search_only()
        print("\n" + "=" * 60)
        print("ПЕРЕХОД К ПОЛНОМУ ТЕСТУ RAG")
        print("=" * 60)
        test_rag_system()
    else:
        print("По умолчанию запускаю полный тест...")
        test_rag_system()
