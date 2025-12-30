"""
центральный менеджер всех модулей (поиск, объединение, роутеры).
"""

from typing import List, Dict, Any, Optional
from src.core.base import BaseSearchModule, BaseFusionModule, BaseRouter
import os
import json


class ModuleManager:
    def __init__(self, storage_path: str = "data/modules"):
        # ВСЕГДА используем cache/modules чтобы избежать Permission denied
        self.storage_path = "cache/modules"
        os.makedirs(self.storage_path, exist_ok=True)
        print(f"ModuleManager: используем путь {self.storage_path}")

        self.search_modules: Dict[str, BaseSearchModule] = {}
        self.fusion_modules: Dict[str, BaseFusionModule] = {}
        self.routers: Dict[str, BaseRouter] = {}

        self.active_searchers: List[str] = []
        self.active_fusion: Optional[str] = None
        self.active_router: Optional[str] = None

        self._load_config()

    def register_search_module(self, module: BaseSearchModule, activate: bool = True):
        """Регистрирует модуль поиска."""
        self.search_modules[module.name] = module

        if activate and module.name not in self.active_searchers:
            self.active_searchers.append(module.name)

        self._save_config()
        return self

    def register_e5_module(
        self,
        name: str = "e5_reranker",
        model_path: str = "./models/e5/e5-small-v2",
        bm25_module_name: str = "bm25",
        top_k_candidates: int = 100,
        device: Optional[str] = None,
        activate: bool = True,
    ):
        """Создает и регистрирует E5 модуль."""
        from src.core.modules.e5_module import E5Module

        # Проверяем существование BM25 модуля
        if bm25_module_name not in self.search_modules:
            print(f"Предупреждение: BM25 модуль '{bm25_module_name}' не найден при создании E5")
            # Можно создать BM25 автоматически или продолжить без него

        print(f"ModuleManager: создаю E5Module '{name}' с storage_path='{self.storage_path}'")

        # Создаем E5 модуль с передачей manager
        e5_module = E5Module(
            name=name,
            model_path=model_path,
            bm25_module_name=bm25_module_name,
            top_k_candidates=top_k_candidates,
            device=device,
            manager=self,
            storage_path=self.storage_path,
        )

        self.register_search_module(e5_module, activate=activate)
        return e5_module

    def register_bm25_module(
        self,
        name: str = "bm25",
        language: str = "russian",
        k1: float = 1.2,
        b: float = 0.75,
        activate: bool = True,
    ):
        """Создает и регистрирует BM25 модуль."""
        from src.core.modules.bm25_module import BM25Module

        print(f"ModuleManager: создаю BM25Module '{name}'")

        bm25_module = BM25Module(name=name, language=language, k1=k1, b=b)

        self.register_search_module(bm25_module, activate=activate)
        return bm25_module

    def unregister_search_module(self, name: str):
        if name in self.search_modules:
            del self.search_modules[name]

        if name in self.active_searchers:
            self.active_searchers.remove(name)

        self._save_config()

    def register_fusion_module(self, name: str, module: BaseFusionModule, activate: bool = False):
        self.fusion_modules[name] = module

        if activate:
            self.active_fusion = name

        self._save_config()

    def register_router(self, name: str, module: BaseRouter, activate: bool = False):
        self.routers[name] = module

        if activate:
            self.active_router = name

        self._save_config()

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        results = {}

        for module_name in self.active_searchers:
            if module_name in self.search_modules:
                module = self.search_modules[module_name]
                try:
                    # Для E5 проверяем загрузку модели
                    if hasattr(module, "model_loaded"):
                        if not getattr(module, "model_loaded", False):
                            results[module_name] = {
                                "status": "skipped",
                                "reason": "model_not_loaded",
                            }
                            continue

                    result = module.add_documents(documents, ids)
                    results[module_name] = result
                except Exception as e:
                    results[module_name] = {"status": "error", "error": str(e)}

        self.save_all()

        return {"status": "success", "details": results, "total_modules": len(results)}

    def search(self, query: str, top_k: int = 5, strategy: str = "auto") -> Dict:
        if strategy == "auto" and self.active_router:
            module_names = self.routers[self.active_router].route(query)
        elif strategy == "all":
            module_names = self.active_searchers.copy()
        elif isinstance(strategy, list):
            module_names = [name for name in strategy if name in self.active_searchers]
        else:
            module_names = self.active_searchers.copy()

        all_results = {}
        for module_name in module_names:
            if module_name in self.search_modules:
                module = self.search_modules[module_name]
                try:
                    # Для E5 модулей передаем дополнительные параметры если нужно
                    results = module.search(query, top_k * 3)
                    all_results[module_name] = results
                except Exception as e:
                    print(f"ошибка в модуле {module_name}: {e}")
                    all_results[module_name] = []

        if self.active_fusion and self.active_fusion in self.fusion_modules:
            fusion_module = self.fusion_modules[self.active_fusion]
            final_results = fusion_module.fuse(all_results, top_k)
        else:
            final_results = self._default_fusion(all_results, top_k)

        return {
            "query": query,
            "strategy": strategy,
            "modules_used": module_names,
            "results": final_results,
            "all_results": all_results if strategy == "debug" else None,
        }

    def _default_fusion(self, all_results: Dict[str, List[Dict]], top_k: int) -> List[Dict]:
        scores = {}

        for module_name, results in all_results.items():
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", f"{module_name}_{rank}")
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_items[:top_k]:
            for results in all_results.values():
                for doc in results:
                    if doc.get("id") == doc_id:
                        final_doc = doc.copy()
                        final_doc["fusion_score"] = float(score)
                        final_results.append(final_doc)
                        break
                if final_results and final_results[-1].get("id") == doc_id:
                    break

        return final_results

    def save_all(self):
        for name, module in self.search_modules.items():
            module.save(self.storage_path)

        self._save_config()

    def load_all(self):
        """Загружает все модули из конфигурации."""
        config = self._load_config()
        if not config:
            print("ModuleManager: конфигурация не найдена, начинаем с чистого листа")
            return False

        print(f"ModuleManager: загружаю конфигурацию из {self.storage_path}")

        # Загружаем модули поиска
        for module_config in config.get("search_modules", []):
            module_type = module_config.get("type")
            module_name = module_config.get("name")
            module_params = module_config.get("params", {})

            # УДАЛЯЕМ name из module_params чтобы не было дублирования
            if "name" in module_params:
                module_params.pop("name")

            try:
                if module_type == "bm25":
                    from src.core.modules.bm25_module import BM25Module

                    # Теперь можно безопасно передавать
                    module = BM25Module(name=module_name, **module_params)

                elif module_type == "e5":
                    from src.core.modules.e5_module import E5Module

                    # Добавляем manager и storage_path
                    module_params["manager"] = self
                    module_params["storage_path"] = self.storage_path
                    module = E5Module(name=module_name, **module_params)

                else:
                    print(f"ModuleManager: неизвестный тип модуля: {module_type}")
                    continue

                # Пытаемся загрузить сохраненные данные
                if module_type == "e5":
                    # Для E5 не передаем путь, пусть сам решит
                    load_result = module.load()
                else:
                    load_result = module.load(self.storage_path)

                # Теперь load_result всегда dict (после исправления BM25Module)
                if isinstance(load_result, dict) and load_result.get("status") in [
                    "loaded",
                    "not_found",
                ]:
                    self.register_search_module(
                        module, activate=module_name in config.get("active_searchers", [])
                    )
                    print(f"ModuleManager: модуль {module_name} ({module_type}) загружен")
                else:
                    print(f"ModuleManager: ошибка загрузки модуля {module_name}: {load_result}")

            except Exception as e:
                print(f"ModuleManager: ошибка создания модуля {module_name}: {e}")
                import traceback

                traceback.print_exc()

        # Восстанавливаем активные модули
        self.active_searchers = config.get("active_searchers", [])
        self.active_fusion = config.get("active_fusion")
        self.active_router = config.get("active_router")

        print(
            f"ModuleManager: загружено {len(self.search_modules)} модулей, активных: {len(self.active_searchers)}"
        )
        return True

    def _save_config(self):
        """Сохраняет конфигурацию системы."""
        config = {
            "active_searchers": self.active_searchers,
            "active_fusion": self.active_fusion,
            "active_router": self.active_router,
            "search_modules": [
                {
                    "name": name,
                    "type": module.get_info().get("type", "unknown"),
                    "params": self._extract_module_params(module),
                }
                for name, module in self.search_modules.items()
            ],
        }

        config_path = os.path.join(self.storage_path, "system_config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"ModuleManager: конфигурация сохранена в {config_path}")
        except Exception as e:
            print(f"ModuleManager: ошибка сохранения конфигурации: {e}")

    def _extract_module_params(self, module) -> Dict:
        """Извлекает параметры модуля для сохранения."""
        info = module.get_info()
        params = {}

        if info.get("type") == "bm25":
            # НЕ сохраняем name - он уже есть в module_config
            params["language"] = info.get("language", "russian")
            params["k1"] = info.get("k1", 2.5)
            params["b"] = info.get("b", 0.9)

        elif info.get("type") == "e5":
            # НЕ сохраняем name
            params["model_path"] = info.get("model", "./models/e5/e5-small-v2")
            params["bm25_module_name"] = info.get("bm25_source", "bm25")
            params["top_k_candidates"] = info.get("top_k_candidates", 100)
            params["device"] = info.get("device", "cpu")

        return params

    def _load_config(self) -> Optional[Dict]:
        config_path = os.path.join(self.storage_path, "system_config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"ModuleManager: ошибка загрузки конфигурации: {e}")
                return None

        return None

    def get_info(self) -> Dict:
        return {
            "total_search_modules": len(self.search_modules),
            "active_search_modules": self.active_searchers,
            "available_fusion_modules": list(self.fusion_modules.keys()),
            "active_fusion": self.active_fusion,
            "available_routers": list(self.routers.keys()),
            "active_router": self.active_router,
            "storage_path": self.storage_path,
            "module_details": {
                name: module.get_info() for name, module in self.search_modules.items()
            },
        }

    def get_module(self, name: str) -> Optional[BaseSearchModule]:
        """Возвращает модуль по имени."""
        return self.search_modules.get(name)
