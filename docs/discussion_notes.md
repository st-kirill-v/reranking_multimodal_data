# Notes по разделу Discussion

## Наиболее важные ограничения

- Оценка проведена на мультимодальном подмножестве DocBench; это релевантный benchmark, но он не покрывает все возможные типы PDF-документов и домены.
- Использованы готовые модели и компоненты, включая Nemotron VL Reranker, ColPali/ColVision и Qwen3-VL; новая архитектура reranker не обучалась.
- Fine-tuning не выполнялся. Это ограничивает потенциальный максимум качества, но повышает воспроизводимость и переносимость pipeline.
- Мультимодальный reranking увеличивает latency, поэтому его применение должно учитывать практические ограничения inference.

## Идеи для защиты проекта

- Главный объект работы - мультимодальный reranking, а не новый retriever или новая VLM.
- Retrieval отвечает за candidate pool, но именно reranking определяет, какое evidence попадёт в генерацию ответа.
- Text reranking полезен как быстрый baseline, но мультимодальный reranking лучше подходит для таблиц, графиков, сложного layout и визуально похожих страниц.
- Практическая ценность работы состоит в понимании quality-latency trade-off для document QA и MRAG pipeline.
- Работа показывает, когда мультимодальный reranking оправдывает свою вычислительную стоимость.

## Наиболее перспективные направления дальнейшей работы

- Обучение специализированного multimodal reranker для выбора evidence в document QA.
- Knowledge distillation для ускорения visual-language reranking.
- Adaptive reranking: выбор между no-reranker, text reranker и multimodal reranker в зависимости от типа вопроса и сложности evidence.
- Расширение оценки на MMLongBench-Doc и другие document QA benchmarks.
- Дополнительная оптимизация latency и inference cost для практического использования в MRAG-системах.
