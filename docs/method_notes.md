# Method Notes

## Связь Method с темой проекта

Официальная тема проекта: **"Разработка алгоритма реранкинга мультимодальных данных"**.

Раздел `4 Method` написан так, чтобы центральным объектом исследования был именно **multimodal reranking**. Pipeline описывает не новый retriever, не новую VLM и не общий multimodal RAG framework, а стратегию переупорядочивания мультимодальных страниц-кандидатов перед генерацией ответа.

## Почему статья позиционируется как работа по multimodal reranking

Методологическая ось раздела:

```text
No Reranker
-> Text Reranker
-> Multimodal Reranker
```

Эта линия показывает, как меняется качество evidence selection при переходе:

- от отсутствия второго этапа ранжирования;
- к text-only reranking;
- к visual-language / multimodal reranking.

Поэтому retriever, evidence construction и VLM рассматриваются не как самостоятельные исследовательские цели, а как условия, влияющие на эффективность reranking.

## Центральные части метода

Центральные элементы:

- `Proposed Multimodal Reranking Strategy`;
- сравнение `No Reranker`, `Text Reranker`, `Multimodal Reranker`;
- Nemotron VL Reranker как основной visual-language reranking component;
- анализ latency cost reranking;
- анализ того, как text evidence и layout-aware crops влияют на reranking.

## Вспомогательные части метода

Вспомогательные компоненты:

- `Candidate Generation`: формирует top-k страницы-кандидаты для reranker.
- `Evidence Construction`: подготавливает full page, layout crop, OCR, page_text, captions и table_text.
- `VLM Answer Generation`: используется как downstream evaluator of reranked evidence.
- `Evaluation Protocol`: измеряет качество ответа, retrieval metrics и latency.

## Ограничения позиционирования

В разделе Method намеренно не утверждается, что работа:

- добавляет отдельный этап обучения reranker-компонента;
- предлагает новую VLM;
- является retrieval benchmark;
- является VLM benchmark;
- является general multimodal RAG study.

Формулировка метода: **multimodal reranking strategy / reranking pipeline for document question answering**.
