"""
центральный менеджер всех модулей (поиск, объединение, роутеры).
"""

from typing import List, Dict, Any, Optional
from src.core.base import BaseSearchModule, BaseFusionModule, BaseRouter
import os
import json


class ModuleManager:
    def __init__(self, storage_path: str = "data/modules"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        self.search_modules: Dict[str, BaseSearchModule] = {}
        self.fusion_modules: Dict[str, BaseFusionModule] = {}
        self.routers: Dict[str, BaseRouter] = {}

        self.active_searchers: List[str] = []
        self.active_fusion: Optional[str] = None
        self.active_router: Optional[str] = None

        self._load_config()

    def register_search_module(self, module: BaseSearchModule, activate: bool = True):
        self.search_modules[module.name] = module

        if activate and module.name not in self.active_searchers:
            self.active_searchers.append(module.name)

        self._save_config()

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
        config = self._load_config()
        if not config:
            return False

        for module_config in config.get("search_modules", []):
            module_type = module_config.get("type")
            module_name = module_config.get("name")

            if module_type == "bm25":
                from src.core.modules.bm25_module import BM25Module

                module = BM25Module(name=module_name)
                if module.load(self.storage_path):
                    self.register_search_module(module, activate=True)

        return True

    def _save_config(self):
        config = {
            "active_searchers": self.active_searchers,
            "active_fusion": self.active_fusion,
            "active_router": self.active_router,
            "search_modules": [
                {"name": name, "type": module.get_info().get("type", "unknown")}
                for name, module in self.search_modules.items()
            ],
        }

        config_path = os.path.join(self.storage_path, "system_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _load_config(self) -> Optional[Dict]:
        config_path = os.path.join(self.storage_path, "system_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def get_info(self) -> Dict:
        return {
            "total_search_modules": len(self.search_modules),
            "active_search_modules": self.active_searchers,
            "available_fusion_modules": list(self.fusion_modules.keys()),
            "active_fusion": self.active_fusion,
            "available_routers": list(self.routers.keys()),
            "active_router": self.active_router,
            "module_details": {
                name: module.get_info() for name, module in self.search_modules.items()
            },
        }
