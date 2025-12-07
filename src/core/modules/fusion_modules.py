from src.core.base import BaseFusionModule
from typing import List, Dict
import numpy as np


class WeightedFusion(BaseFusionModule):
    """Взвешенное объединение результатов"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}

    def fuse(self, all_results: Dict[str, List[Dict]], top_k: int = 5) -> List[Dict]:
        scores = {}

        for module_name, results in all_results.items():
            weight = self.weights.get(module_name, 1.0)

            for rank, doc in enumerate(results):
                doc_id = doc.get("id", f"{module_name}_{rank}")
                # Взвешенный RRF
                scores[doc_id] = scores.get(doc_id, 0) + weight / (60 + rank + 1)

        return self._get_top_results(scores, all_results, top_k)

    def _get_top_results(self, scores, all_results, top_k):
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


class RRFusion(BaseFusionModule):
    """Reciprocal Rank Fusion (стандартный)"""

    def fuse(self, all_results: Dict[str, List[Dict]], top_k: int = 5) -> List[Dict]:
        scores = {}

        for module_name, results in all_results.items():
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", f"{module_name}_{rank}")
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

        return WeightedFusion()._get_top_results(scores, all_results, top_k)
