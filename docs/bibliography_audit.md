# Bibliography Audit

Дата проверки: 2026-06-18.

Проверялись источники из `docs/related_work.md`, раздел `References`. Основная проверка выполнялась по страницам arXiv, указанным в списке литературы. Для всех источников подтверждено наличие arXiv-страницы и arXiv DOI формата `10.48550/arXiv.<id>`.

## Краткий итог

- Всего проверено источников: 21.
- Источники существуют: 21 / 21.
- Дубликаты в списке `References`: не обнаружены.
- Источники с пометкой `arXiv preprint` или `ongoing/work in progress`: есть, их следует явно оставлять как preprint, если не указан peer-reviewed venue.
- Основная потенциальная правка, найденная в ходе аудита, была связана с источником [20]: точное название на arXiv начинается с `M3-Embedding`, а не `BGE M3-Embedding`. Название в `docs/related_work.md` исправлено.

## Проверка по источникам

| Ref | Статус | Год | Авторы | Название | DOI / arXiv | Комментарий |
| --- | --- | ---: | --- | --- | --- | --- |
| [1] | OK | 2020 / WACV 2021 | Minesh Mathew, Dimosthenis Karatzas, C. V. Jawahar | `DocVQA: A Dataset for VQA on Document Images` | arXiv:2007.00398, DOI: 10.48550/arXiv.2007.00398 | Год arXiv 2020, venue WACV 2021. Авторы и название корректны. |
| [2] | OK | 2022 | Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei | `LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking` | arXiv:2204.08387, DOI: 10.48550/arXiv.2204.08387 | ACM Multimedia 2022 указан на arXiv. |
| [3] | OK | 2021 / ECCV 2022 | Geewook Kim et al. | `OCR-free Document Understanding Transformer` | arXiv:2111.15664, DOI: 10.48550/arXiv.2111.15664 | В списке литературы используется `et al.`, первый автор корректен. |
| [4] | OK | 2022 / ICML 2023 | Kenton Lee et al. | `Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding` | arXiv:2210.03347, DOI: 10.48550/arXiv.2210.03347 | Год arXiv 2022, accepted at ICML. |
| [5] | OK | 2024 | Yubo Ma et al. | `MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations` | arXiv:2407.01523, DOI: 10.48550/arXiv.2407.01523 | NeurIPS 2024 Datasets and Benchmarks Track, Spotlight. |
| [6] | OK / preprint | 2024 | Anni Zou et al. | `DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems` | arXiv:2407.10701, DOI: 10.48550/arXiv.2407.10701 | На arXiv указано `Work in progress`; корректно помечать как preprint / benchmark paper. |
| [7] | OK | 2021 | Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych | `BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models` | arXiv:2104.08663, DOI: 10.48550/arXiv.2104.08663 | Название на arXiv содержит `Heterogenous`; список авторов корректен. |
| [8] | OK | 2024 / ICLR 2025 | Manuel Faysse et al. | `ColPali: Efficient Document Retrieval with Vision Language Models` | arXiv:2407.01449, DOI: 10.48550/arXiv.2407.01449 | Год arXiv 2024; в таблице можно оставить указание ICLR 2025. |
| [9] | OK | 2024 / ICLR 2025 | Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, Wei Ping | `MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs` | arXiv:2411.02571, DOI: 10.48550/arXiv.2411.02571 | Accepted at ICLR 2025. |
| [10] | OK / preprint | 2025 | Mengyao Xu et al. | `Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video` | arXiv:2510.03458, DOI: 10.48550/arXiv.2510.03458 | Источник существует; по состоянию на проверку это arXiv preprint. |
| [11] | OK / preprint | 2026 | António Loison et al. | `ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios` | arXiv:2601.08620, DOI: 10.48550/arXiv.2601.08620 | Источник существует; использовать как актуальный preprint / benchmark paper. |
| [12] | OK / ongoing preprint | 2023 | Yunfan Gao et al. | `Retrieval-Augmented Generation for Large Language Models: A Survey` | arXiv:2312.10997, DOI: 10.48550/arXiv.2312.10997 | На arXiv указано `Ongoing Work`; корректно считать survey preprint. |
| [13] | OK | 2024 | Wenqi Fan et al. | `A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models` | arXiv:2405.06211, DOI: 10.48550/arXiv.2405.06211 | Long version of KDD 2024 survey paper. Не является дубликатом [12], это отдельный survey. |
| [14] | OK | 2022 | Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, William W. Cohen | `MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text` | arXiv:2210.02928, DOI: 10.48550/arXiv.2210.02928 | Accepted to EMNLP 2022. |
| [15] | OK / preprint | 2025 | Zhenghao Liu et al. | `Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts` | arXiv:2502.17297, DOI: 10.48550/arXiv.2502.17297 | В тексте может называться M2RAG benchmark, но в списке литературы лучше оставить точное arXiv-название. |
| [16] | OK / preprint | 2025 | Ziyu Gong, Chengcheng Mai, Yihua Huang | `MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning` | arXiv:2508.00579, DOI: 10.48550/arXiv.2508.00579 | Источник существует; авторы в таблице 3.6 указаны корректно. |
| [17] | OK / preprint | 2025 | Zirui Guo et al. | `RAG-Anything: All-in-One RAG Framework` | arXiv:2510.12323, DOI: 10.48550/arXiv.2510.12323 | Источник существует; следует помечать как arXiv preprint. |
| [18] | OK | 2020 | Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin | `Document Ranking with a Pretrained Sequence-to-Sequence Model` | arXiv:2003.06713, DOI: 10.48550/arXiv.2003.06713 | MonoT5-related citation, авторы и название корректны. |
| [19] | OK | 2020 | Omar Khattab, Matei Zaharia | `ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT` | arXiv:2004.12832, DOI: 10.48550/arXiv.2004.12832 | Accepted at SIGIR 2020. |
| [20] | OK после правки | 2024 | Jianlv Chen et al. | `M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation` | arXiv:2402.03216, DOI: 10.48550/arXiv.2402.03216 | Точное arXiv-название внесено в `docs/related_work.md`. |
| [21] | OK / preprint | 2026 | Mingxin Li et al. | `Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking` | arXiv:2601.04720, DOI: 10.48550/arXiv.2601.04720 | Источник существует. В abstract arXiv есть внутренняя формулировка `as of January 8, 2025`, хотя submission датирован 2026; это выглядит как ошибка в самом preprint, не в нашем списке. |

## Дубликаты

Явных дубликатов в списке `References` не обнаружено.

Пары, которые могут выглядеть похожими, но не являются дубликатами:

- [12] и [13] - разные survey papers по RAG / RA-LLM.
- [8] и [11] - связаны с visual document retrieval / ViDoRe, но это разные работы.
- [9], [10], [21] - все относятся к multimodal retrieval/reranking, но имеют разные arXiv IDs, авторов и постановки.

## Рекомендации перед финальной версией статьи

1. Для [6], [10], [11], [12], [15], [16], [17], [21] сохранить пометки `arXiv preprint`, `benchmark paper`, `work in progress` или аналогичные, если они используются в таблицах related work.
2. Если требуется строгий библиографический стиль, заменить `et al.` в финальном списке литературы на полный список авторов или использовать BibTeX, экспортированный с arXiv.
3. Оставить [21] как релевантный future/related work source, но не привязывать его к проведённым экспериментам, поскольку он опубликован после основных экспериментальных запусков.
