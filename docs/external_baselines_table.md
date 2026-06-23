# External Baselines Table

Дата обзора: 2026-06-23.

## Main Table

| Work | Dataset | Retrieval | Reranking | Fine-tuning | Metric | Result |
| --- | --- | --- | --- | --- | --- | --- |
| DOCBENCH | DocBench | Не изолирован | Нет | Нет | Accuracy | Human 81.2; KimiChat 70.9; GPT-4 document system 69.8; parse-then-read GPT-4 67.9 |
| M3DocRAG | M3DocVQA, MMLongBench-Doc, MP-DocVQA | Да, multimodal retriever | Evidence selection через retrieval | Использует trained/pretrained retrieval+MLM components | ANLS / benchmark metrics | Reports superior performance and SOTA on MP-DocVQA |
| ColPali | ViDoRe | Да, visual page retrieval | Late interaction scoring | Да | Retrieval metrics | Outperforms modern document retrieval pipelines |
| MM-Embed | M-BEIR, MTEB, CIRCO | Да, universal multimodal retrieval | Да, zero-shot MLLM reranking | Да | M-BEIR score, MTEB, mAP@5 | SOTA on M-BEIR; +7 mAP@5 on CIRCO with MLLM reranking |
| MuRAG | WebQA, MultimodalQA | Да, multimodal memory | Нет | Да | Accuracy | +10-20% absolute over previous systems |
| M2RAG / MM-RAIT | M2RAG benchmark | Да | Да, image reranking task | Да | Task-specific RAG metrics | +34% / +33% gains over MiniCPM-V 2.6 and Qwen2-VL |
| VisRAG | Multi-modality document datasets | Да, vision-based retrieval | Evidence selection via retrieval | Да, retriever training | End-to-end performance | 20-40% gain over traditional text-based RAG |
| Enhancing Document VQA via RAG | MP-DocVQA, DUDE, InfographicVQA | Да, OCR text + visual retrieval | Да | Нет / not training-centric | ANLS | Up to +22.5 ANLS for text-centric RAG; +5.0 ANLS for visual RAG |
| CMRAG | Visual document VQA tasks | Да, text + image channels | Cross-modal aggregation | Не указано | VQA metrics | Significantly outperforms pure-vision RAG |
| ViDoRAG | ViDoSeek | Да, GMM hybrid multimodal retrieval | Agentic evidence selection | Не указано | Dataset score | Over 10% improvement over existing methods |
| ViDoRe V3 | ViDoRe v3, 10 datasets | Да, textual and visual retrievers | Да, textual reranking / late interaction | Benchmark paper | Retrieval, answer quality, grounding | Visual retrievers outperform textual; late interaction and textual reranking improve performance |
| MMRAG-DocQA | MMLongBench-Doc, LongDocURL | Да, hierarchical multi-granularity retrieval | Да, LLM-based reranking | Не central | Dataset metrics | Reports superiority over baselines |
| MAGE-RAG | LongDocURL, MMLongBench-Doc | Да, page retrieval + evidence graph | Evidence graph controller | Нет / not central | Accuracy, F1 | LongDocURL 52.75 Acc; MMLongBench-Doc 53.26 Acc / 51.19 F1 |
| DocDancer | MMLongBench-Doc, DocBench | Да, search/read tools | Agentic evidence selection | Да, 5k trajectories | LasJ / Acc / F1 | DocBench LasJ: GPT-5.2 85.5; Qwen3-30B ft 81.2 |
| MARDoc | MMLongBench-Doc, DocBench | Да, Explorer retrieval | Refiner/Reflector memory | Нет task-specific training | LasJ / Acc / F1 | DocBench LasJ: Qwen3-VL-30B 82.1; Qwen3-VL-8B 72.3 |
| Qwen3-VL-Embedding / Reranker | MMEB-V2 and multimodal retrieval/ranking benchmarks | Да | Да, cross-encoder multimodal reranker | Да | MMEB-V2 | Qwen3-VL-Embedding-8B: 77.8 overall |
| Document Haystacks / V-RAG | DocHaystack, InfoHaystack | Да, vision-centric retrieval over 1000+ document images | Dedicated relevance module | Не указано | Recall@1 | +9% Recall@1 on DocHaystack-1000; +11% Recall@1 on InfoHaystack-1000 |

## Groups for Positioning

### A. Вероятно ниже нашего уровня для нашей конкретной постановки

| Work | Reason |
| --- | --- |
| DOCBENCH parse-then-read / upload-system baselines | Нет controlled reranking; не анализируют multimodal evidence selection |
| OCR-only / text-only RAG baselines | В большинстве document RAG работ уступают visual or multimodal retrieval |
| Naive VLM over all pages | Нет retrieval/reranking; высокая стоимость и слабая localization |
| Pure-vision-only RAG baselines | В CMRAG и других работах уступают co-modality или hybrid approaches |

### B. Сопоставимый уровень / ближайшие внешние baseline

| Work | Reason |
| --- | --- |
| Enhancing Document VQA via RAG | Retrieval + reranking gains on MP-DocVQA / DUDE / InfographicVQA |
| M3DocRAG | Multimodal retriever + VLM answer generation |
| MMRAG-DocQA | Retrieval + LLM-based reranking + long document QA |
| ViDoRAG | Hybrid multimodal retrieval + iterative evidence selection |
| ViDoRe V3 | Directly evaluates retrieval, reranking, generation and grounding |
| MM-Embed | Multimodal retrieval and MLLM reranking |
| Qwen3-VL-Reranker | Direct multimodal reranking model baseline |

### C. Существенно сильнее или масштабнее, но не напрямую сопоставимо

| Work | Reason |
| --- | --- |
| DocDancer | Agentic training and very high DocBench LasJ; not reranking-isolation study |
| MARDoc | Strong DocBench LasJ with structured memory and Qwen3-VL |
| ViDoRe V3 | Larger benchmark scope and grounding annotations |
| MAGE-RAG | Graph/agentic evidence construction over long documents |
| Qwen3-VL-Embedding / Reranker | Specialized model series for multimodal retrieval/ranking |

## Recommended External Baselines for Discussion

| Priority | Work | Why include |
| ---: | --- | --- |
| 1 | DOCBENCH | Dataset source and official document reading baseline |
| 2 | Enhancing Document VQA via RAG | Explicit retrieval + reranking evidence for Document VQA |
| 3 | M3DocRAG | Strong multimodal RAG over multi-page documents |
| 4 | MMRAG-DocQA | Hierarchical retrieval and LLM-based reranking |
| 5 | ViDoRe V3 | Shows visual retrieval and reranking matter for end-to-end multimodal RAG |
| 6 | ColPali | Visual document retrieval foundation |
| 7 | MM-Embed | Multimodal retrieval + MLLM reranking |
| 8 | Qwen3-VL-Reranker | Model-level multimodal reranking reference |
| 9 | DocDancer | Strong DocBench agentic baseline |
| 10 | MARDoc | Strong DocBench memory/evidence refinement baseline |
