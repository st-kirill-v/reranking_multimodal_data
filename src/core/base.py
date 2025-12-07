from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import os


class BaseSearchModule(ABC):
    """Абстрактный базовый класс для ЛЮБОГО поискового модуля"""

    @abstractmethod
    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None) -> Dict:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        pass

    @abstractmethod
    def clear(self) -> Dict:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        pass


class BaseFusionModule(ABC):
    """Абстрактный класс для объединения результатов"""

    @abstractmethod
    def fuse(self, all_results: Dict[str, List[Dict]], top_k: int = 5) -> List[Dict]:
        pass


class BaseRouter(ABC):
    """Абстрактный класс для роутинга запросов"""

    @abstractmethod
    def route(self, query: str, document_type: Optional[str] = None) -> List[str]:
        """Возвращает список модулей для использования"""
        pass
