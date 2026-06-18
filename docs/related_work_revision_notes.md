# Related Work Revision Notes

## Что было изменено

- Усилен блок **Research Gap** в конце раздела `3.4 Retrieval Reranking`.
- В `3.5 Positioning of Our Work` добавлен вводный абзац о недостатке systematic component-level evaluation в multimodal document QA.
- В конце `3.5` добавлен блок contributions с четырьмя основными вкладами работы.
- Основная таблица `3.6 Comparison with Existing Work` сокращена до наиболее близких работ.
- После основной таблицы добавлена компактная positioning-таблица по признакам:
  - Visual Retrieval;
  - Reranking;
  - Multimodal QA;
  - DocBench;
  - Latency Analysis.
- Улучшены переходы между reranking literature, research gap и позиционированием нашей работы.
- Исправлено название работы `arXiv:2508.00579`: вместо `MMRAG-DocQA` указано актуальное название `MHier-RAG`.

## Статьи, удалённые из сравнительной таблицы 3.6

Эти работы оставлены в тексте разделов и/или списке литературы, но удалены из большой сравнительной таблицы, чтобы сделать её компактнее:

- DocVQA;
- LayoutLMv3;
- Donut;
- Pix2Struct;
- MMLongBench-Doc;
- BEIR;
- Omni-Embed-Nemotron;
- ViDoRe v3;
- M2RAG;
- MonoT5;
- ColBERT.

Причина удаления: эти работы важны для общего контекста Related Work, но менее близки к нашему итоговому сравнению, чем работы по visual retrieval, multimodal RAG, document QA reranking и DocBench.

## Статьи, оставленные в основной таблице 3.6

- DocBench;
- ColPali;
- MM-Embed;
- MuRAG;
- RAG-Anything;
- MHier-RAG;
- BGE M3;
- Qwen3-VL-Embedding / Qwen3-VL-Reranker;
- Наша работа.

Эти работы оставлены, потому что они напрямую связаны с одной или несколькими ключевыми частями нашего исследования: benchmark, visual retrieval, multimodal retrieval, multimodal RAG, reranking, text embedding baseline или multimodal reranking.

## Проверка источников

Для работ, оставленных в таблицах, были проверены arXiv-страницы:

- `DocBench`: arXiv:2407.10701, 2024, preprint.
- `ColPali`: arXiv:2407.01449, 2024; на arXiv указано, что работа опубликована как conference paper at ICLR 2025.
- `MM-Embed`: arXiv:2411.02571, 2024, preprint.
- `MuRAG`: arXiv:2210.02928, 2022, preprint.
- `RAG-Anything`: arXiv:2510.12323, 2025, preprint.
- `MHier-RAG`: arXiv:2508.00579, 2025, preprint.
- `BGE M3`: arXiv:2402.03216, 2024, preprint.
- `Qwen3-VL-Embedding / Qwen3-VL-Reranker`: arXiv:2601.04720, 2026, preprint.

## Потенциальные проблемы

- `Qwen3-VL-Embedding / Qwen3-VL-Reranker` относится к 2026 году и опубликована после основных экспериментов проекта, поэтому в тексте она обозначена как future direction, а не как компонент текущих экспериментов.
- `RAG-Anything` и `Qwen3-VL-Embedding / Qwen3-VL-Reranker` являются очень свежими preprint-ами; их статус следует перепроверить перед финальной подачей статьи.
- Работа `arXiv:2508.00579` ранее была указана как `MMRAG-DocQA`, но актуальная arXiv-страница содержит название `MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning`.
