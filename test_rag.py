# Простой тест в Python
from src.core.rag import rag_engine

# Тестовый запрос
query = "Что такое искусственный интеллект?"

# Поиск через E5 (он внутри использует BM25)
results = rag_engine.manager.search_modules["e5_reranker"].search(query, top_k=5)

print(f"🔍 Запрос: {query}")
print(f"📊 Найдено результатов: {len(results)}")

for i, doc in enumerate(results[:3], 1):
    print(f"\n{i}. {doc.get('content', '')[:200]}...")
    print(f"   Score: {doc.get('score'):.3f}")
    print(f"   BM25 score: {doc.get('bm25_score', 0):.3f}")
    print(f"   E5 score: {doc.get('e5_score', 0):.3f}")
