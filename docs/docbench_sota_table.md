# DocBench SOTA Table

Дата обзора: 2026-06-23.

## Основная таблица

| Paper | DocBench | RAG | Reranking | Fine-Tuning | Best Metric |
| --- | --- | --- | --- | --- | --- |
| DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems | Полный датасет: 229 PDF, 1,102 questions | Нет как отдельный MRAG-framework; оцениваются document reading systems и parse-then-read pipeline | Нет | Нет | Human 81.2 Acc; KimiChat 70.9 Acc; GPT-4 document system 69.8 Acc; parse-then-read GPT-4 67.9 Acc |
| DocDancer: Towards Agentic Document-Grounded Information Seeking | Используется; статья указывает 229 documents и 1,082 questions | Да, agentic information-seeking с search/read tools; сравнение с RAG-based baselines | Нет standalone reranker | Да, Qwen3-30B-A3B-Thinking и Qwen3-4B-Thinking fine-tuned на 5,000 agent trajectories | DocBench LasJ: GPT-5.2 85.5; Qwen3-30B-A3B ft 81.2; Qwen3-4B ft 79.8 |
| MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA | Используется; 229 documents и 1,102 questions | Да, iterative multimodal retrieval-reasoning через Explorer/Refiner/Reflector | Нет standalone reranker; retrieval направляется memory/reflection loop | Нет task-specific training | DocBench LasJ: Qwen3-VL-30B-A3B-Instruct 82.1; Qwen3-VL-8B-Instruct 72.3 |

## Расширенная таблица по компонентам

| Paper | Year | Task | Models | Retrieval | Reranking | Evaluation |
| --- | ---: | --- | --- | --- | --- | --- |
| DOCBENCH | 2024 | LLM-based document reading | GPT-4, GPT-4o, Claude-3, KimiChat, GLM-4, Qwen-2.5, ERNIE-3.5, open LLMs | Не изолирован | Нет | GPT-4 evaluator, Accuracy |
| DocDancer | 2026 | Agentic document-grounded information seeking | GPT-4o, Gemini-2.5-Pro, GPT-5.2, Qwen3-4B/30B Thinking | Search/read tools over document outline | Нет | Official DocBench judge prompts via GPT-4.1 |
| MARDoc | 2026 | Memory-aware multimodal long-document QA | Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-8B-Instruct | Explorer retrieves multi-granularity textual/visual evidence | Нет | Official DocBench judge prompts via GPT-4o |

## Работы, которые не следует считать прямыми DocBench papers

| Paper / Method | Причина |
| --- | --- |
| VisRAG | Релевантен для multimodal RAG, но прямое использование DocBench в найденном источнике не подтверждено |
| ColPali | Релевантен для visual document retrieval, но оценивается на ViDoRe, не на DocBench |
| SV-RAG | Релевантен для long-document RAG, но прямое использование DocBench в найденном источнике не подтверждено |
| RAGAnything | General multimodal RAG framework; появляется как baseline в DocDancer/MARDoc, но не подтверждён как оригинальная DocBench evaluation |
| BookRAG | Появляется как baseline в MARDoc; прямое использование DocBench в оригинальном найденном источнике не подтверждено |
| Dr. DocBench | Отдельный benchmark, не DocBench Zou et al. |
| MMDocBench | Отдельный benchmark, не DocBench Zou et al. |

## Короткий вывод для нашей статьи

Существующий SOTA на DocBench быстро смещается от simple document reading и parse-then-read pipelines к agentic frameworks с search/read tools, structured memory и reflection. Однако найденные работы не изолируют мультимодальный reranking как центральный компонент. Поэтому наша работа занимает отдельную нишу: controlled experimental study of multimodal reranking for DocBench-based Document QA.
