from src.core.module_manager import ModuleManager
from src.core.modules.bm25_module import BM25Module
from src.core.modules.fusion_modules import RRFusion
from src.core.modules.router_modules import DebugRouter
from typing import List, Dict, Any, Optional
import torch


class ModularRAG:
    """Главный RAG движок с модульной архитектурой"""

    def __init__(self, storage_path: str = "data/modules"):
        self.manager = ModuleManager(storage_path)
        self._index_built = False

        # Инициализируем стандартные модули
        self._init_default_modules()

        # Загружаем сохраненное состояние
        self.manager.load_all()

    def _init_default_modules(self):
        """
        Инициализация с каскадным BM25→E5
        """
        print("🚀 Инициализация RAG системы с каскадным поиском BM25→E5...")

        # 1. BM25 модуль (основа для каскада)
        bm25_module = BM25Module(name="bm25", language="multilingual")
        self.manager.register_search_module(bm25_module, activate=True)  # АКТИВЕН
        print("   ✅ BM25 модуль: загружен")

        # 2. E5 модуль (каскадный с BM25)
        try:
            from src.core.modules.e5_module import E5Module

            e5_module = E5Module(
                name="e5_reranker",
                model_path="./models/e5/e5-small-v2",
                bm25_module_name="bm25",  # будет использовать этот BM25
                top_k_candidates=100,
                model_name="intfloat/multilingual-e5-small",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.manager.register_search_module(e5_module, activate=True)  # АКТИВЕН
            print("   ✅ E5 модуль: загружен (каскадный с BM25)")
        except ImportError as e:
            print(f"   ⚠️  E5 модуль: ошибка - {e}")
            print("       Установите: pip install transformers torch")

        # 3. Fusion модуль (если нужно объединять с другими модулями)
        rrf = RRFusion()
        self.manager.register_fusion_module("rrf", rrf, activate=True)
        print("   ✅ Fusion модуль: RRF")

        # 4. Роутер
        router = DebugRouter()
        self.manager.register_router("smart", router, activate=True)
        print("   ✅ Роутер: DebugRouter")

        print("\n🎯 Система готова! Архитектура: BM25 → E5 (каскадный)")

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        """Добавить документы"""
        return self.manager.add_documents(documents, ids)

    def search(self, query: str, n_results: int = 5, strategy: str = "auto") -> Dict:
        """Поиск с улучшенной нормализацией scores"""
        # Выполняем поиск через manager
        result = self.manager.search(query, n_results, strategy)

        # Форматируем результаты
        formatted_results = []
        for doc in result["results"]:
            formatted_results.append(
                {
                    "id": doc.get("id"),
                    "document": doc.get("content", ""),
                    "score": doc.get("fusion_score", doc.get("score", 0.0)),
                    "module": doc.get("module", "unknown"),
                }
            )

        # 🔥 ШАГ 3: ГЛОБАЛЬНАЯ НОРМАЛИЗАЦИЯ И BOOSTING
        if formatted_results:
            # 1. Собираем все скоры
            all_scores = [doc["score"] for doc in formatted_results]

            # 2. Нормализация только если есть различия
            if len(set(all_scores)) > 1:  # Есть разные значения
                max_score = max(all_scores)
                min_score = min(all_scores)

                if max_score > min_score:
                    # 3. Применяем нормализацию и boosting
                    for doc in formatted_results:
                        orig_score = doc["score"]

                        # Min-max нормализация
                        norm_score = (orig_score - min_score) / (max_score - min_score)

                        # 🔥 Boosting на основе длины документа
                        doc_length = len(doc["document"].split())

                        if 15 <= doc_length <= 150:  # Идеальная длина для ответа
                            norm_score *= 1.3  # +30% boost
                        elif doc_length < 10:  # Слишком короткий
                            norm_score *= 0.7  # -30% penalty
                        elif doc_length > 300:  # Слишком длинный
                            norm_score *= 0.8  # -20% penalty

                        # Гарантируем границы [0, 1]
                        doc["score"] = min(1.0, max(0.0, norm_score))

            # 4. Сортируем по новым скорам
            formatted_results.sort(key=lambda x: x["score"], reverse=True)

            # 5. Ограничиваем количество результатов
            formatted_results = formatted_results[:n_results]

        return {
            "query": query,
            "results": formatted_results,
            "normalized": True,  # Флаг что нормализация применена
        }

    def get_info(self) -> Dict:
        """Информация о системе"""
        return self.manager.get_info()

    def clear_documents(self) -> Dict:
        """Очистить все документы"""
        for name, module in self.manager.search_modules.items():
            module.clear()

        self.manager.save_all()
        return {"status": "cleared"}

    def load_index(self) -> bool:
        """Загрузить индекс"""
        return self.manager.load_all()

    def build_index(self) -> Dict[str, Any]:
        """🔥 ИСПРАВЛЕННЫЙ МЕТОД: Строит индексы для всех модулей"""
        print("🔨 Начинаю построение индексов...")

        results = {}

        # 🔥 ИСПРАВЛЕНИЕ: active_searchers содержит имена модулей (строки), а не объекты!
        for module_name in self.manager.active_searchers:
            print(f"  📝 Обрабатываю модуль: {module_name}")

            # Получаем реальный объект модуля
            if module_name not in self.manager.search_modules:
                print(f"    ❌ Модуль '{module_name}' не найден в search_modules")
                results[module_name] = {"status": "error", "message": "Module not found"}
                continue

            module = self.manager.search_modules[module_name]

            try:
                # 🔥 ИСПРАВЛЕНИЕ: Получаем документы из manager
                if hasattr(self.manager, "documents") and self.manager.documents:
                    documents = self.manager.documents
                    print(f"    📚 Документов в manager: {len(documents)}")
                elif hasattr(module, "documents") and module.documents:
                    documents = module.documents
                    print(f"    📚 Документов в модуле: {len(documents)}")
                else:
                    print(f"    ⚠️ Нет документов для модуля {module_name}")
                    results[module_name] = {"status": "error", "message": "No documents"}
                    continue

                # Обучение модуля
                if hasattr(module, "fit"):
                    print(f"    🎯 Вызываю fit()...")
                    result = module.fit(documents)
                    results[module_name] = {
                        "status": "success",
                        "method": "fit",
                        "documents": len(documents),
                    }
                elif hasattr(module, "add_documents"):
                    print(f"    📥 Вызываю add_documents()...")
                    result = module.add_documents(documents)
                    results[module_name] = {
                        "status": "success",
                        "method": "add_documents",
                        "documents": len(documents),
                    }
                else:
                    results[module_name] = {"status": "skipped", "message": "No indexing method"}

                print(f"    ✅ {module_name}: {results[module_name]['status']}")

            except Exception as e:
                print(f"    ❌ {module_name}: ошибка - {str(e)}")
                results[module_name] = {"status": "error", "message": str(e)}

        self._index_built = True
        self.manager.save_all()

        print(f"✅ Построение индексов завершено")
        return {
            "status": "success",
            "message": "Index rebuilt",
            "details": {
                "status": "completed",
                "index_built": True,
                "total_modules": len(results),
                "results": results,
            },
        }

    def is_index_built(self) -> bool:
        """Проверить построен ли индекс"""
        return self._index_built

    def get_document_count(self) -> int:
        """Получить количество документов"""
        if hasattr(self.manager, "documents"):
            return len(self.manager.documents)
        return 0

    def add_search_module(self, module_type: str, name: str, **kwargs) -> Dict:
        """Добавить новый поисковый модуль"""
        if module_type == "e5":  # ← НОВАЯ ВЕТКА ДЛЯ E5
            from src.core.modules.e5_module import E5Module

            module = E5Module(name=name, **kwargs)  # Создаем E5 модуль
            self.manager.register_search_module(module, activate=True)
            return {"status": "added", "name": name, "type": "e5"}

        elif module_type == "clip":  # ← существующая ветка
            from src.core.modules.clip_module import CLIPModule

            module = CLIPModule(name=name, **kwargs)

        elif module_type == "layoutlm":  # ← существующая ветка
            from src.core.modules.layoutlm_module import LayoutLMModule

            module = LayoutLMModule(name=name, **kwargs)

        else:  # ← если неизвестный тип модуля
            return {"status": "error", "message": f"Unknown module type: {module_type}"}

        # Для существующих модулей (clip, layoutlm) регистрация здесь:
        self.manager.register_search_module(module, activate=True)
        return {"status": "added", "name": name, "type": module_type}

    def remove_search_module(self, name: str) -> Dict:
        """Удалить поисковый модуль"""
        self.manager.unregister_search_module(name)
        return {"status": "removed", "name": name}

    def list_modules(self) -> Dict:
        """Список всех модулей"""
        return self.manager.get_info()


# Глобальный экземпляр
rag_engine = ModularRAG()
