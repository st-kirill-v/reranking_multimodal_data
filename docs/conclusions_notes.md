# Notes по разделу Conclusions

## Главные выводы

- Document QA по PDF-документам требует качественного выбора страниц и evidence перед генерацией ответа.
- Разработана и реализована мультимодальная стратегия reranking, интегрированная в document QA pipeline.
- Reranking улучшает качество по сравнению с retrieval-only pipeline.
- Multimodal reranking превосходит text-only reranking на мультимодальном подмножестве DocBench.
- Визуальная генерация кандидатов и `full page + layout crop` создают наиболее сильные условия для эффективного reranking.
- Улучшение качества достигается ценой latency, поэтому итоговый выбор конфигурации зависит от прикладного quality-latency trade-off.

## Выводы для Abstract

- Работа посвящена development and evaluation of a multimodal reranking strategy for document question answering.
- Центральное сравнение статьи: `No Reranker -> Text Reranker -> Multimodal Reranker`.
- Эксперименты показывают, что multimodal reranking даёт более высокое качество answer generation, чем text-only reranking, когда вопросы требуют визуального или layout-aware evidence.
- Практический вклад работы состоит в анализе quality-latency trade-off для reranking внутри воспроизводимого document QA pipeline.

## Выводы для защиты проекта

- Главный объект разработки - не новая VLM и не новый retriever, а мультимодальный reranking pipeline.
- Reranking отвечает за выбор страниц, crops и evidence, которые затем получает модель генерации ответа.
- Результаты показывают, что мультимодальный reranking особенно полезен для вопросов, зависящих от таблиц, layout, визуальных областей и похожих страниц-кандидатов.
- Text reranking остаётся быстрым baseline, но не достигает качества мультимодального reranking на мультимодальном DocBench subset.
- Основное ограничение мультимодального reranking - latency, поэтому дальнейшая работа должна быть направлена на ускорение, distillation и adaptive reranking.
