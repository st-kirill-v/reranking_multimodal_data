# External Baselines Review

Дата обзора: 2026-06-23.

## Цель обзора

Этот документ собирает внешние работы по направлениям:

- Multimodal RAG;
- Document QA / PDF Question Answering;
- Visual Document Retrieval;
- Multimodal Retrieval;
- Multimodal Reranking;
- Long Document QA;
- evidence selection для VLM/LLM answer generation.

Фокус сделан на pipeline-ах вида:

```text
Question
-> Retrieval
-> Reranking / Evidence Selection
-> VLM or LLM
-> Answer
```

Важно: результаты ниже не сравниваются напрямую с нашими числами, если используются разные датасеты и метрики. В обзоре отдельно указано, где используются Accuracy, F1, EM, ANLS, Recall, LasJ или другие показатели.

## Краткая карта работ

| Work | Year | Dataset | Retrieval | Reranking / Evidence Selection | Fine-tuning | Metric / Result |
| --- | ---: | --- | --- | --- | --- | --- |
| DOCBENCH | 2024 | DocBench | Не изолирован | Нет отдельного reranker | Нет | Accuracy: Human 81.2; KimiChat 70.9; GPT-4 document system 69.8; parse-then-read GPT-4 67.9 |
| M3DocRAG | 2024 | M3DocVQA, MMLongBench-Doc, MP-DocVQA | Да, multimodal retriever | Evidence selection через retrieval; standalone reranker не указан | Да/используются pretrained ColPali и Qwen2-VL; отдельный fine-tuning framework не является главным акцентом | Сообщает superior performance и SOTA на MP-DocVQA |
| ColPali | 2024 / ICLR 2025 | ViDoRe | Да, visual page retrieval | Late interaction как retrieval scoring; не QA reranking | Да, VLM trained for multi-vector page embeddings | Retrieval metrics на ViDoRe; outperforms modern document retrieval pipelines |
| MM-Embed | 2024 | M-BEIR, MTEB, CIRCO | Да, universal multimodal retrieval | Да, zero-shot MLLM reranking | Да, MLLM bi-encoder fine-tuning + hard negatives | SOTA on M-BEIR; zero-shot MLLM reranking improves CIRCO by >7 mAP@5 |
| MuRAG | 2022 | WebQA, MultimodalQA | Да, multimodal memory retrieval | Нет отдельного reranker | Да, joint contrastive + generative pretraining | +10-20% absolute over previous systems |
| M2RAG / MM-RAIT | 2025 | M2RAG benchmark: image captioning, multimodal QA, fact verification, image reranking | Да, open-domain multimodal retrieval | Есть task image reranking; context utilization через instruction tuning | Да, MM-RAIT | +34% / +33% gains over MiniCPM-V 2.6 and Qwen2-VL in reported RAG settings |
| VisRAG | 2024/2025 | Multi-modality documents | Да, vision-based retrieval | Evidence selection через visual retrieval | Да, retriever training | 20-40% end-to-end gain over traditional text-based RAG |
| Enhancing Document VQA via RAG | 2025 | MP-DocVQA, DUDE, InfographicVQA | Да, text OCR retrieval and visual retrieval | Да, ablation says retrieval and reranking drive most gain | Не является training-centric; evaluated variants across models | Up to +22.5 ANLS for text-centric RAG; +5.0 ANLS for visual RAG |
| CMRAG | 2025 | Visual document VQA tasks | Да, text + image candidate evidence retrieval | Cross-modal aggregation; explicit reranker not central | Не указано в abstract | Significantly outperforms pure-vision RAG |
| ViDoRAG | 2025 | ViDoSeek | Да, GMM-based hybrid multimodal retrieval | Agentic exploration / summarization / reflection as evidence selection | Не указан как main fine-tuning result | Outperforms existing methods by over 10% on ViDoSeek |
| ViDoRe V3 | 2026 | ViDoRe v3, 10 datasets, 3,099 queries | Да, textual and visual retrievers | Да, textual reranking and late interaction analyzed | Benchmark/evaluation paper | Finds visual retrievers outperform textual ones; late-interaction and textual reranking improve performance |
| MMRAG-DocQA | 2025 | MMLongBench-Doc, LongDocURL | Да, hierarchical index and multi-granularity retrieval | Да, LLM-based reranking | Не является fine-tuning-centric | Reports superiority over baselines on public long-document QA datasets |
| MAGE-RAG | 2026 | LongDocURL, MMLongBench-Doc | Да, page retrieval + graph evidence activation | Evidence controller opens/searches/prunes evidence graph | Нет/не центрально | LongDocURL 52.75 Acc; MMLongBench-Doc 53.26 Acc / 51.19 F1 |
| DocDancer | 2026 | MMLongBench-Doc, DocBench | Да, search/read tools | Agentic evidence selection; no standalone reranker | Да, Qwen3 4B/30B fine-tuned on 5k trajectories | DocBench LasJ: GPT-5.2 85.5; Qwen3-30B ft 81.2 |
| MARDoc | 2026 | MMLongBench-Doc, DocBench | Да, Explorer retrieves multi-granularity evidence | Refiner/Reflector evidence memory; no standalone reranker | Нет task-specific training | DocBench LasJ: Qwen3-VL-30B 82.1; Qwen3-VL-8B 72.3 |
| Qwen3-VL-Embedding / Reranker | 2026 | MMEB-V2 and multimodal retrieval/ranking benchmarks | Да, embedding model | Да, cross-encoder multimodal reranker | Да, multi-stage training and distillation | Qwen3-VL-Embedding-8B: 77.8 overall on MMEB-V2 |

## Notes by Work

### DOCBENCH

**DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems** introduces the benchmark used in our work. It evaluates proprietary document-reading systems and parse-then-read pipelines on 229 PDF documents and 1,102 questions. It does not isolate retrieval or reranking, but it defines the realistic document QA setting and official accuracy-style evaluation.

Use for our paper:

- dataset motivation;
- question-type taxonomy;
- evidence localization difficulty;
- baseline for document reading systems rather than reranking.

### M3DocRAG

**M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding** proposes a multimodal RAG framework for single-document and multi-document DocVQA. It uses a multimodal retriever and an MLM, with experiments on M3DocVQA, MMLongBench-Doc and MP-DocVQA. The paper reports superior performance and SOTA on MP-DocVQA.

Use for our paper:

- closest prior for multimodal retrieval + VLM answer generation;
- useful external baseline for visual retrieval and multi-page evidence selection;
- differs from us because it emphasizes multimodal retrieval, while our central object is reranking.

### ColPali

**ColPali: Efficient Document Retrieval with Vision Language Models** is a key visual document retrieval work. It directly embeds document page images and uses late interaction. It is not a Document QA paper by itself, but it is highly relevant because ColPali-style retrieval often acts as the first stage before reranking or VLM generation.

Use for our paper:

- visual page retrieval baseline;
- candidate generation stage;
- explanation why visual retrieval can be stronger than OCR-only retrieval on rich PDFs.

### MM-Embed

**MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs** studies MLLM-based multimodal retrieval and zero-shot MLLM reranking. It reports SOTA on M-BEIR and more than 7 mAP@5 improvement from MLLM reranking on composed image retrieval.

Use for our paper:

- strong support for the idea that MLLMs can act as rerankers;
- closest general multimodal retrieval/reranking paper;
- differs because it is not Document QA over PDFs and does not evaluate final answer generation from DocBench-style evidence.

### MuRAG

**MuRAG** is an early multimodal retrieval-augmented generator. It uses an external multimodal memory and joint contrastive/generative training, reporting 10-20% absolute improvements on WebQA and MultimodalQA.

Use for our paper:

- historical MRAG baseline;
- establishes retrieval-augmented generation with image+text memory;
- less close than document-specific works because it does not target visual PDF page evidence.

### M2RAG / MM-RAIT

**Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts** introduces the M2RAG benchmark and MM-RAIT instruction tuning. It includes image captioning, multimodal QA, multimodal fact verification and image reranking.

Use for our paper:

- shows multimodal context utilization and reranking as an explicit benchmark task;
- useful for related work on multimodal retrieval/reranking;
- differs because it is open-domain multimodal RAG, not PDF Document QA.

### VisRAG

**VisRAG** proposes vision-based RAG for multimodality documents: pages are embedded as images and retrieved for VLM generation. It reports 20-40% end-to-end performance gain over text-based RAG.

Use for our paper:

- visual RAG external baseline;
- supports using page images directly;
- differs because reranking is not central.

### Enhancing Document VQA Models via RAG

This work is especially close to our evidence-selection motivation. It evaluates RAG for Document VQA on MP-DocVQA, DUDE and InfographicVQA. It considers text-based OCR retrieval and visual retrieval. The authors report up to +22.5 ANLS for text-centric RAG and +5.0 ANLS for visual RAG; they also state that retrieval and reranking components drive most of the gain.

Use for our paper:

- closest external support for “retrieval + reranking/evidence selection improves Document VQA”;
- useful for discussion even though metrics are ANLS, not Mean F1;
- differs because it does not focus on DocBench and does not isolate multimodal VL reranking as the main object.

### CMRAG

**CMRAG: Co-modality-based document retrieval and visual question answering** retrieves candidate evidence from text and image channels and aggregates results at the cross-modal retrieval level before prompting the VLM.

Use for our paper:

- external support for combining text and image evidence;
- relevant to our fusion configurations;
- less direct because no standalone reranker is reported as the central component.

### ViDoRAG

**ViDoRAG** introduces ViDoSeek and a multi-agent RAG framework for visually rich documents. It uses GMM-based hybrid multimodal retrieval and iterative exploration/summarization/reflection. It reports over 10% improvement on ViDoSeek.

Use for our paper:

- comparable agentic evidence-selection direction;
- useful for future work: iterative reranking/evidence refinement;
- not directly comparable to our Mean F1 because dataset and metric differ.

### ViDoRe V3

**ViDoRe V3** is an end-to-end multimodal RAG benchmark with retrieval relevance, bounding boxes and verified answers. It covers about 26,000 pages and 3,099 queries in 6 languages. The evaluation finds that visual retrievers outperform textual ones and that late-interaction models plus textual reranking improve performance.

Use for our paper:

- very strong benchmark reference for visual document retrieval and multimodal RAG;
- supports our use of visual candidate generation and reranking;
- not directly comparable because it is benchmark/evaluation focused and not DocBench.

### MMRAG-DocQA

**MMRAG-DocQA** uses hierarchical indexing, multi-granularity retrieval and LLM-based reranking for long multimodal documents. It is evaluated on MMLongBench-Doc and LongDocURL.

Use for our paper:

- close conceptual baseline because it has retrieval plus LLM-based reranking;
- supports discussion of cross-page fragmentation and inter-modal disconnection;
- differs because it studies hierarchical RAG framework, while our work studies reranking ablations.

### MAGE-RAG

**MAGE-RAG** builds an adaptive graph evidence framework and uses query-time evidence activation/open/search/prune operations before LVLM generation. It reports 52.75 Accuracy on LongDocURL and 53.26 Accuracy / 51.19 F1 on MMLongBench-Doc.

Use for our paper:

- strong future-work direction for graph-based evidence selection;
- comparable to our layout-aware evidence construction, but more agentic/graph-based;
- not directly comparable to our DocBench Mean F1.

### DocDancer

**DocDancer** is an agentic document-grounded information seeking framework with search/read tools and synthetic trajectory training. It uses DocBench and MMLongBench-Doc. It fine-tunes Qwen3-4B and Qwen3-30B variants on 5,000 trajectories and reports strong DocBench LasJ results.

Use for our paper:

- strongest DocBench external baseline;
- demonstrates that agentic evidence search can exceed human baseline under their LasJ setting;
- differs because it trains agent behavior and does not isolate reranking.

### MARDoc

**MARDoc** uses Explorer, Refiner and Reflector agents to maintain structured evidence memory for multimodal long-document QA. It uses DocBench and MMLongBench-Doc and reports 82.1 DocBench LasJ with Qwen3-VL-30B.

Use for our paper:

- closest DocBench work using Qwen3-VL and evidence refinement;
- strong discussion point for memory-aware evidence selection;
- differs because it is an agentic memory framework, not a controlled reranking study.

### Qwen3-VL-Embedding / Qwen3-VL-Reranker

This report introduces Qwen3-VL embedding and reranker models for text, image, document image and video inputs. Qwen3-VL-Reranker is a cross-encoder for fine-grained relevance estimation; Qwen3-VL-Embedding-8B reports 77.8 on MMEB-V2.

Use for our paper:

- future replacement for Nemotron VL reranker;
- direct relevance to multimodal reranking;
- not a Document QA paper, but a model-level baseline.

## A. Работы, где результат вероятно ниже нашего уровня

Эта группа не означает прямое численное превосходство нашей Mean F1 над чужой Accuracy/ANLS/Recall. Здесь собраны работы или baseline-ы, которые по своей постановке ближе к более простому evidence selection или уступают современным agentic/multimodal methods в собственных таблицах.

| Work | Почему вероятно ниже / менее близко |
| --- | --- |
| DOCBENCH parse-then-read baselines | Нет controlled retrieval/reranking; сильные ошибки на multimodal и metadata questions |
| Text-only / OCR-only RAG baselines в M3DocRAG, ViDoRe V3, DocDancer/MARDoc | Обычно хуже visual или multimodal retrieval settings в соответствующих работах |
| Naive VLM / concatenate-all-pages approaches | Высокая стоимость контекста и слабая evidence localization |
| Basic RAG baselines типа VisRAG/ColPali when used without agentic refinement in DocDancer/MARDoc tables | В более новых agentic работах уступают DocAgent/DocDancer/MARDoc |
| Pure-vision RAG in CMRAG comparison | CMRAG сообщает преимущество co-modality retrieval over pure-vision RAG |

## B. Работы сопоставимого уровня

Эти работы близки по исследовательской постановке: retrieval/evidence selection перед VLM/LLM answer generation, визуальные PDF-документы, multi-page QA, RAG или reranking.

| Work | Почему сопоставимо |
| --- | --- |
| Enhancing Document VQA Models via RAG | Прямо показывает вклад retrieval and reranking components в Document VQA |
| M3DocRAG | Multi-modal RAG для MP-DocVQA/MMLongBench-Doc с ColPali + Qwen2-VL |
| MMRAG-DocQA | Имеет hierarchical retrieval и LLM-based reranking |
| ViDoRAG | Multi-agent visual document RAG с hybrid retrieval и reflection |
| ViDoRe V3 | Benchmark-level работа, анализирует visual retrievers, reranking and generation |
| CMRAG | Text+image retrieval aggregation перед VLM answer |
| MM-Embed | Multimodal retrieval + zero-shot MLLM reranking, но не Document QA |
| Qwen3-VL-Reranker | Сильная model-level работа по multimodal reranking |

## C. Работы существенно сильнее нашего уровня

Здесь указаны работы, которые могут быть сильнее не по прямой метрике, а по масштабу модели, agentic complexity, training regime или reported benchmark SOTA. Их нельзя напрямую сравнивать с нашим Mean F1 на DocBench multimodal subset.

| Work | Почему сильнее / что учитывать |
| --- | --- |
| DocDancer | Сильные DocBench LasJ результаты; использует agentic training on 5k trajectories и closed/proprietary backbones in some variants |
| MARDoc | На DocBench LasJ достигает 82.1 с Qwen3-VL-30B; использует structured memory, reflection and multi-granularity retrieval |
| ViDoRe V3 | Масштабный benchmark с 26k pages, 3,099 queries, grounding boxes and multilingual evaluation |
| MAGE-RAG | Более сложная graph/agentic evidence construction framework for long-document QA |
| Qwen3-VL-Embedding / Reranker | Специализированная модельная линейка для multimodal retrieval and reranking; потенциально сильнее Nemotron-style reranking для future work |

## Potential External Baselines for Our Paper

Для статьи лучше выбрать не все найденные работы, а 5-10 наиболее близких:

1. **DOCBENCH** - основной dataset source и baseline document reading systems.
2. **Enhancing Document VQA Models via RAG** - ближайшая работа про Document VQA, retrieval and reranking gains.
3. **M3DocRAG** - сильный multimodal RAG baseline for multi-page / multi-document DocVQA.
4. **MMRAG-DocQA** - hierarchical index + multi-granularity retrieval + LLM reranking.
5. **ViDoRe V3** - benchmark-level evidence that visual retrievers and reranking improve multimodal RAG.
6. **ColPali** - foundational visual document retrieval baseline.
7. **MM-Embed** - general multimodal retrieval and MLLM reranking evidence.
8. **Qwen3-VL-Embedding / Qwen3-VL-Reranker** - direct model-level multimodal reranking reference.
9. **DocDancer** - strong agentic DocBench baseline, useful to show what our work does not attempt.
10. **MARDoc** - closest DocBench + Qwen3-VL agentic evidence refinement baseline.

Recommended positioning:

> Existing works either build general multimodal RAG frameworks, visual retrieval benchmarks, or agentic long-document QA systems. Our work is narrower and more controlled: it isolates the reranking stage inside a DocBench-based Document QA pipeline and compares No Reranker, Text Reranker and Multimodal Reranker under shared evidence and VLM settings.

## Sources

1. Anni Zou et al. **DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems**. https://arxiv.org/abs/2407.10701
2. Jaemin Cho et al. **M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding**. https://arxiv.org/abs/2411.04952
3. Manuel Faysse et al. **ColPali: Efficient Document Retrieval with Vision Language Models**. https://arxiv.org/abs/2407.01449
4. Sheng-Chieh Lin et al. **MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs**. https://arxiv.org/abs/2411.02571
5. Wenhu Chen et al. **MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text**. https://arxiv.org/abs/2210.02928
6. Zhenghao Liu et al. **Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts**. https://arxiv.org/abs/2502.17297
7. Shi Yu et al. **VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents**. https://arxiv.org/abs/2410.10594
8. Eric López et al. **Enhancing Document VQA Models via Retrieval-Augmented Generation**. https://arxiv.org/abs/2508.18984
9. Wang Chen et al. **CMRAG: Co-modality-based document retrieval and visual question answering**. https://arxiv.org/abs/2509.02123
10. Qiuchen Wang et al. **ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents**. https://arxiv.org/abs/2502.18017
11. António Loison et al. **ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios**. https://arxiv.org/abs/2601.08620
12. Ziyu Gong et al. **MMRAG-DocQA: A Multi-Modal Retrieval-Augmented Generation Method for Document Question-Answering with Hierarchical Index and Multi-Granularity Retrieval**. https://arxiv.org/abs/2508.00579
13. Yilong Zuo et al. **MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA**. https://arxiv.org/abs/2606.15906
14. Qintong Zhang et al. **DocDancer: Towards Agentic Document-Grounded Information Seeking**. https://arxiv.org/abs/2601.05163
15. Kaifeng Chen et al. **MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA**. https://arxiv.org/abs/2606.05749
16. Mingxin Li et al. **Qwen3-VL-Embedding and Qwen3-VL-Reranker**. https://arxiv.org/abs/2601.04720
17. Jun Chen et al. **Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents**. https://arxiv.org/abs/2411.16740
