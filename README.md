# Тема: Разработка алгоритма реранкинга мультимодальных данных

> Исследование посвящено разработке и оценке алгоритма **реранкинга мультимодальных данных** для задач Document Question Answering.

Основная задача проекта — определить, как мультимодальный реранкинг помогает выбирать релевантные текстовые и визуальные evidence из PDF-документов. Retrieval, evidence construction и VLM используются как элементы экспериментального контура, но центральным объектом исследования остаётся именно этап реранкинга.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![Dataset](https://img.shields.io/badge/Dataset-DocBench-green)](#dataset)
[![Task](https://img.shields.io/badge/Task-Multimodal%20Reranking-purple)](#project-overview)
[![Paper](https://img.shields.io/badge/Paper-article__final.md-orange)](article_final.md)

---

## Навигация

| Раздел | Ссылка |
| --- | --- |
| О проекте | [Project Overview](#project-overview) |
| Pipeline | [Pipeline](#pipeline) |
| Данные и модели | [Dataset & Models](#dataset--models) |
| Результаты | [Results](#results) |
| Запуск | [Quick Start](#quick-start) |
| Воспроизведение | [Reproducing](#reproducing) |
| Документация | [Documentation](#documentation) |

---

<a id="project-overview"></a>

## Project Overview

**Document Question Answering** — это задача ответа на вопросы по документам. Для PDF-документов ответ может зависеть не только от текста, но и от таблиц, графиков, изображений страниц и layout-структуры.

В этом проекте исследуется не общий Multimodal RAG и не сравнение VLM, а именно **мультимодальный реранкинг**: этап, который переупорядочивает найденные страницы-кандидаты и выбирает evidence для генерации ответа.

Основной научный вклад:

- построен воспроизводимый Document QA pipeline;
- проведено сравнение `No Reranker`, `Text Reranker` и `Multimodal Reranker`;
- показано, что мультимодальный реранкинг улучшает качество, но увеличивает latency;
- определены сильные конфигурации для quality-oriented и speed-oriented сценариев.

---

<a id="pipeline"></a>

## Pipeline

```text
Question
   ↓
Candidate Generation
   ↓
Multimodal Reranking
   ↓
Evidence Construction
   ↓
Qwen3-VL
   ↓
Answer
```

В pipeline используются:

- **retrievers:** Nemotron, ColPali / ColVision, BM25, BGE;
- **rerankers:** Nemotron VL Reranker, BGE, Jina, MiniLM;
- **VLM:** Qwen3-VL-30B, Qwen3-VL-8B;
- **evidence:** page text, OCR, captions, table text, full page, layout crop.

Retrieval и VLM рассматриваются как компоненты, влияющие на эффективность реранкинга, а не как самостоятельный объект исследования.

---

<a id="dataset--models"></a>

## Dataset & Models

Эксперименты выполнены на мультимодальном подмножестве **DocBench**.

| Параметр | Значение |
| --- | ---: |
| PDF-документы | 229 |
| Вопросы в полном DocBench | 1,102 |
| Используемые multimodal questions | 308 |
| Типы вопросов | `multimodal-t`, `multimodal-f` |

| Component | Models |
| --- | --- |
| Retrievers | Nemotron image retrieval, ColPali / ColVision, BM25, BGE text encoders |
| Rerankers | Nemotron VL Reranker, BGE-reranker-base, BGE-reranker-large, Jina, MiniLM |
| VLM | Qwen3-VL-30B, Qwen3-VL-8B |

---

<a id="results"></a>

## Results

Ключевые результаты из итоговой статьи:

| Результат | Конфигурация | Mean F1 | Latency |
| --- | --- | ---: | ---: |
| Лучшее качество | Nemotron full image+text + VL reranker + Qwen3-VL-30B | **0.7023** | 13.6441 sec |
| Лучший быстрый вариант | Fusion Nemotron no image reranker + Qwen3-VL-30B | **0.6575** | **2.5080 sec** |
| Лучший text-reranker baseline | BM25 + BGE-reranker-large + Qwen3-VL-30B | 0.5497 | 1.2344 sec |

| Итог | Значение |
| --- | ---: |
| Средний прирост от reranking | **+0.0428 Mean F1** |
| Средняя стоимость по latency | **+5.8355 sec** |

Основные выводы:

- реранкинг стабильно улучшает качество Document QA;
- мультимодальный реранкинг превосходит текстовый реранкинг;
- Nemotron retrieval формирует наиболее сильные candidate pools;
- `full page + layout crop` является наиболее эффективной evidence strategy;
- рост качества сопровождается увеличением latency.

Полные таблицы:

- [paper_multimodal_308.md](reports/tables/paper_multimodal_308.md)
- [paper_multimodal_308.csv](reports/tables/paper_multimodal_308.csv)
- [experiment summary](reports/experiment_summary/)

---

<a id="quick-start"></a>

## Quick Start

```bash
git clone https://github.com/st-kirill-v/reranking_multimodal_data.git
cd reranking_multimodal_data
```

Через `uv`:

```bash
uv venv
uv sync
```

Windows PowerShell:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync
```

Для экспериментов с Qwen3-VL через OpenAI-compatible backend:

```powershell
$env:OPENAI_COMPAT_API_KEY="..."
```

---

<a id="reproducing"></a>

## Reproducing

Основной запуск выполняется через YAML-конфиги:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/image_text_full_308_nemotron_qwen3vl30b.yaml
```

No-reranker baseline:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/image_text_full_308_nemotron_no_reranker_qwen3vl30b.yaml
```

Text-reranker baseline:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/text_reranker_308_bge_large_qwen3vl30b.yaml
```

Агрегация результатов:

```bash
python scripts/build_experiment_summary_tables.py
```

Ожидаемые артефакты:

```text
reports/tables/paper_multimodal_308.csv
reports/tables/paper_multimodal_308.md
reports/experiment_summary/
```

Подробная методика и экспериментальная постановка описаны в [article_final.md](article_final.md).

---

<a id="documentation"></a>

## Documentation

| Документ | Описание |
| --- | --- |
| [article_final.md](article_final.md) | Итоговая статья на русском языке |
| [docs/](docs/) | Разделы статьи, аудит, публикационный план |
| [reports/tables/](reports/tables/) | Таблицы результатов |
| [reports/experiment_summary/](reports/experiment_summary/) | Агрегированные результаты |
| [paper_ieee/](paper_ieee/) | IEEE Conference версия статьи |

---

## Citation

```bibtex
@misc{stulov2026multimodalreranking,
  title        = {Разработка алгоритма реранкинга мультимодальных данных},
  author       = {Стулов, Кирилл Вячеславович},
  year         = {2026},
  institution  = {Университет ИТМО}
}
```

---

## Authors

**Автор:** Стулов Кирилл Вячеславович
**Научный руководитель:** Вершинин Владислав Константинович
**Организация:** Университет ИТМО
**Направление:** 09.04.02 Информационные системы и технологии
