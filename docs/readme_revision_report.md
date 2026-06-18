# README Revision Report

Дата: 2026-06-19.

## Что было изменено

- Полностью заменён старый `README.md`, который содержал устаревшее описание проекта и отображался с mojibake в текущей консоли.
- README перепозиционирован в соответствии с итоговой статьёй: центральный объект исследования — **Multimodal Reranking**.
- Retrieval, evidence construction и VLM описаны как вспомогательные компоненты, влияющие на эффективность реранкинга.
- Добавлены GitHub-friendly структура, бейджи, кликабельное оглавление, таблицы, ссылки на локальные артефакты и команды запуска.
- Добавлен отдельный блок `Быстрая навигация` с гиперссылками на основные разделы README.
- Добавлены дополнительные локальные ссылки внутри разделов `Экспериментальные результаты`, `Запуск экспериментов` и `Документация проекта`.
- После дополнительной редакции README сокращён до компактной GitHub landing page: оставлены только обзор проекта, pipeline, датасет/модели, ключевые результаты, быстрый запуск, воспроизведение и ссылки на документацию.
- Подробные RQ-таблицы, расширенное дерево репозитория, длинные инструкции индексации и TODO-блоки по графикам удалены из README, так как они дублируют статью и отчёты.
- Заголовок README изменён на формат `Тема: Разработка алгоритма реранкинга мультимодальных данных`.
- Первые предложения README переписаны так, чтобы сразу отражать суть работы: разработку и оценку алгоритма реранкинга мультимодальных данных, а не просто описание Document QA pipeline.

## Что добавлено

- Header с названием проекта и кратким описанием.
- `Быстрая навигация` с отдельными гиперссылками на ключевые разделы.
- `Project Overview` с объяснением Document QA, Multimodal RAG и Multimodal Reranking.
- `Research Goal` с целью, гипотезой и исследовательскими вопросами.
- `Pipeline Overview` с ASCII-схемой.
- `Repository Structure` с описанием основных директорий.
- `Dataset` с характеристиками DocBench и используемого multimodal subset.
- `Models` с таблицей retrievers, rerankers, VLM и evidence strategies.
- `Experimental Results` с ключевыми результатами из статьи.
- `Key Findings` с основными выводами исследования.
- `Dashboards & Visualizations` со ссылками на реальные CSV/Markdown-артефакты и TODO по отсутствующим изображениям.
- `Installation`, `Running Experiments`, `Reproducing Results`.
- `Project Documentation`, `Future Work`, `Citation`, `Authors`.
- В финальной компактной версии сохранены только наиболее полезные для первого просмотра разделы: `Project Overview`, `Pipeline`, `Dataset & Models`, `Results`, `Quick Start`, `Reproducing`, `Documentation`, `Citation`, `Authors`.

## Данные, взятые из статьи

Основной источник истины:

- `article_final.md`
- `docs/abstract.md`
- `docs/introduction.md`
- `docs/method.md`
- `docs/experiments.md`
- `docs/discussion.md`
- `docs/conclusions.md`

Использованные численные результаты:

- Best quality configuration: `Nemotron full image+text + VL reranker + Qwen3-VL-30B`, Mean F1 `0.7023`, latency `13.6441 sec`.
- Best fast configuration: `Fusion Nemotron no image reranker + Qwen3-VL-30B`, Mean F1 `0.6575`, latency `2.5080 sec`.
- Best text reranker baseline: `BM25 + BGE-reranker-large + Qwen3-VL-30B`, Mean F1 `0.5497`, latency `1.2344 sec`.
- Average reranker gain: `+0.0428 Mean F1`.
- Average latency cost: `+5.8355 sec`.
- Dataset subset: `308` multimodal DocBench questions.
- Full DocBench description: `229` PDF documents and `1,102` questions.

## Изображения и визуализации

В текущем checkout не обнаружены готовые PNG/SVG/JPG изображения в `reports/`.

README использует ссылки на реальные табличные артефакты:

- `reports/tables/paper_multimodal_308.md`
- `reports/tables/paper_multimodal_308.csv`
- `reports/experiment_summary/main_table.csv`
- `reports/experiment_summary/component_aggregation.csv`
- `reports/experiment_summary/paper_sections/5_3_retriever_analysis.csv`
- `reports/experiment_summary/paper_sections/5_4_reranker_ablation.csv`
- `reports/experiment_summary/paper_sections/5_4_reranker_aggregation.csv`
- `reports/experiment_summary/paper_sections/5_7_quality_speed_recommendations.csv`
- `reports/experiment_summary/paper_sections/paper_section_findings.md`

В README добавлен TODO-блок для будущих изображений:

- Top Experiments.
- Quality vs Latency.
- Reranker Ablation.
- Retriever Comparison.

## Соответствие README разделам статьи

| README section | Article source |
| --- | --- |
| Header | `docs/abstract.md`, `article_final.md` |
| Project Overview | `docs/abstract.md`, `docs/introduction.md` |
| Research Goal | `docs/introduction.md`, `docs/method.md` |
| Pipeline Overview | `docs/method.md` |
| Dataset | `docs/method.md`, `docs/experiments.md` |
| Models | `docs/method.md`, `docs/experiments.md` |
| Experimental Results | `docs/experiments.md`, `reports/tables/paper_multimodal_308.md` |
| Key Findings | `docs/experiments.md`, `docs/discussion.md`, `docs/conclusions.md` |
| Dashboards & Visualizations | `reports/experiment_summary/paper_sections/` |
| Installation | `pyproject.toml`, current script/config structure |
| Running Experiments | `configs/experiments/`, `scripts/run_experiment.py` |
| Reproducing Results | `docs/method.md`, `docs/experiments.md`, scripts in `scripts/` |
| Project Documentation | `docs/`, `article_final.md`, `paper_ieee/` |
| Future Work | `docs/discussion.md`, `docs/conclusions.md` |
| Citation | Project title and author metadata |
| Authors | User-provided project metadata |

## Проверка научной позиции

README не позиционирует проект как:

- общий обзор Multimodal RAG;
- benchmark retriever-ов;
- сравнение VLM как самостоятельную цель.

README позиционирует проект как исследование мультимодального реранкинга в Document QA pipeline. Retrieval, evidence construction и VLM описаны только как компоненты, влияющие на эффективность реранкинга.
