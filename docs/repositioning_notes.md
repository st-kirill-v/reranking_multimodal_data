# Repositioning Notes

## Что было изменено

- `docs/introduction.md` дополнен явной гипотезой о том, что мультимодальный reranking является ключевым компонентом document QA pipeline.
- В `2.3 Research Gap` усилена формулировка о том, что visual retrieval и multimodal RAG уже активно развиваются, но роль multimodal reranking изучена слабее.
- В `2.3` добавлено явное сравнение text-only reranking и multimodal reranking under identical retrieval conditions.
- В `docs/related_work.md` усилен раздел `3.5 Positioning of Our Work`: добавлен тезис, что работа рассматривает multimodal reranking как центральный объект исследования.
- В таблицу `3.6 Comparison with Existing Work` добавлен столбец `Primary Focus`.
- В `docs/paper_materials.md` подраздел `Итоговая архитектура пайплайна` переименован в `Proposed Multimodal Reranking Strategy`.
- В `docs/paper_materials.md` добавлено явное описание предлагаемого метода как мультимодальной reranking strategy.
- В `docs/paper_materials.md` дополнительно зафиксирована главная экспериментальная линия: `No Reranker -> Text Reranker -> Multimodal Reranker`.

## Какие формулировки усилены

- Research Gap теперь подчёркивает не общий multimodal RAG, а недостаточную изученность multimodal reranking в document QA.
- Contributions теперь начинаются с формулировки `Development and evaluation of a multimodal reranking strategy for document question answering`.
- Method positioning теперь описывает retriever, evidence construction и VLM backbone как окружение, влияющее на эффективность reranking.
- Experiments narrative теперь строится вокруг сравнения no-reranker, text-reranker и multimodal-reranker режимов.

## Соответствие теме проекта

Официальная тема проекта: **"Разработка алгоритма реранкинга мультимодальных данных"**.

После правок статья позиционируется именно как исследование мультимодального reranking:

- центральный объект исследования: `Multimodal Reranking`;
- baseline: `No Reranker`;
- промежуточный baseline: `Text Reranker`;
- основной исследуемый режим: `Multimodal Reranker`;
- retriever рассматривается как источник candidates для reranking;
- evidence construction рассматривается как способ обогащения входа reranker;
- VLM backbone рассматривается как downstream evaluator выбранного reranker evidence;
- latency analysis рассматривается как практическое ограничение применения multimodal reranking.

Экспериментальные результаты, таблицы, графики, значения Mean F1 и latency не изменялись.
