# deep_test_bm25.py
import sys

sys.path.append(".")
from src.core.rag import rag_engine

print("=" * 60)
print("🔍 ГЛУБОКАЯ ДИАГНОСТИКА BM25")
print("=" * 60)

bm25 = rag_engine.manager.search_modules["bm25"]

print("\n1. 📦 БАЗОВАЯ ИНФОРМАЦИЯ:")
print(f"   Тип: {type(bm25).__name__}")
print(f"   Имя: {bm25.name}")
print(f"   Язык: {bm25.language}")
print(f"   is_fitted: {bm25.is_fitted}")
print(f"   Документов: {len(bm25.documents)}")
print(f"   Объект bm25: {bm25.bm25}")

print("\n2. 🔧 МЕТОДЫ И АТРИБУТЫ:")
methods = [m for m in dir(bm25) if not m.startswith("_")]
print(f"   Все методы: {methods}")

print("\n3. 🧪 ТЕСТ ПРЕДОБРАБОТКИ:")
test_texts = ["технология", "technology", "World War II", "искусственный интеллект"]

for text in test_texts:
    try:
        tokens = bm25._preprocess_text(text)
        print(f"   '{text}' -> {tokens} (len={len(tokens)})")
    except Exception as e:
        print(f"   ❌ '{text}' -> Ошибка: {e}")

print("\n4. 🧪 ТЕСТ СТОП-СЛОВ:")
if hasattr(bm25, "stop_words"):
    test_words = ["технология", "the", "и", "war", "python"]
    for word in test_words:
        is_stop = word in bm25.stop_words
        print(f"   '{word}' в стоп-словах? {is_stop}")

print("\n5. 🧪 ТЕСТ add_documents С 1 ДОКУМЕНТОМ:")
try:
    # Очищаем
    if hasattr(bm25, "clear"):
        bm25.clear()
        print("   ✅ Очистили BM25")

    # Добавляем 1 простой документ
    test_doc = ["Технология блокчейн используется в Bitcoin"]
    print(f"   Добавляю документ: '{test_doc[0]}'")

    result = bm25.add_documents(test_doc)
    print(f"   Результат add_documents: {result}")
    print(f"   is_fitted после: {bm25.is_fitted}")
    print(f"   Документов теперь: {len(bm25.documents)}")
    print(f"   Объект bm25 создан? {bm25.bm25 is not None}")

except Exception as e:
    print(f"   ❌ Ошибка add_documents: {e}")
    import traceback

    traceback.print_exc()

print("\n6. 🧪 ТЕСТ ПОИСКА (если is_fitted=True):")
if bm25.is_fitted and bm25.bm25 is not None:
    test_queries = ["технология", "blockchain", "биткоин"]
    for query in test_queries:
        try:
            results = bm25.search(query, top_k=2)
            print(f"   Запрос '{query}': {len(results)} результатов")
            if results:
                print(f"     Первый: {results[0]['content'][:50]}...")
        except Exception as e:
            print(f"   ❌ Ошибка поиска '{query}': {e}")
else:
    print(f"   ⚠️ Поиск невозможен: is_fitted={bm25.is_fitted}, bm25={bm25.bm25}")

print("\n7. 🧪 ПРЯМОЙ ВЫЗОВ BM25Okapi:")
try:
    from rank_bm25 import BM25Okapi

    # Тестируем напрямую
    test_docs = ["technology blockchain", "artificial intelligence", "python programming"]
    tokenized_docs = [doc.split() for doc in test_docs]
    print(f"   Тестовые документы: {test_docs}")
    print(f"   Токенизированные: {tokenized_docs}")

    bm25_test = BM25Okapi(tokenized_docs)
    scores = bm25_test.get_scores(["technology"])
    print(f"   BM25Okapi работает! Скоры: {scores}")

except Exception as e:
    print(f"   ❌ Ошибка BM25Okapi: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Диагностика завершена")
print("=" * 60)
