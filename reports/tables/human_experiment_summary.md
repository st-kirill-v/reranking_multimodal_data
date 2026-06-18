# Human Experiment Summary

## Short Conclusions

- Best overall: **Nemotron full image+text + VL reranker + Qwen30B**, mean F1 = **0.7023**, latency = **13.6441s**.
- Fastest visual branch: **Nemotron image no reranker + Qwen30B**, mean F1 = **0.6479**, latency = **2.5535s**.
- Best text branch: **Nemotron full image+text + VL reranker + Qwen30B**, mean F1 = **0.7023**.
- Visual reranking remains important: removing it is much faster, but quality drops.
- Text-only and text-evidence branches are useful advisor baselines, not the best final method.

## Main Table

| Rank | Experiment | Family | Retriever | Reranker | Evidence | VLM | Mean F1 | F1>0.5 | MM-T F1 | MM-F F1 | Latency, s | Comment | Results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Nemotron full image+text + VL reranker + Qwen30B | Full image+text | Nemotron image | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.7023 | 0.7565 | 0.7201 | 0.6578 | 13.6441 | Best overall result; text is used in reranker and VLM input. | results/image_text_full_308_nemotron_qwen3vl30b |
| 2 | Nemotron image+text input + VL reranker + Qwen30B | Image+text input | Nemotron image | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.6979 | 0.7468 | 0.7138 | 0.6579 | 12.9920 | Previous best; text is added to VLM input. | results/image_text_input_308_nemotron_qwen3vl30b |
| 3 | Nemotron image + VL reranker + Qwen30B | Nemotron image retriever | Nemotron image | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.6902 | 0.7500 | 0.7015 | 0.6620 | 11.0702 | Former best image-only result; strong baseline. | results/multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b |
| 4 | ColPali + VL reranker + Qwen30B | ColPali visual Qwen30B | ColPali/ColVision | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.6860 | 0.7338 | 0.6971 | 0.6582 | 14.8791 | Reference baseline. | results/docbench_hybrid_bm25_tools_qwen3vl30b |
| 5 | Fusion Nemotron + VL reranker + Qwen30B | Image+text fusion | Nemotron image | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.6793 | 0.7175 | 0.6933 | 0.6445 | 9.4122 | Best fusion result; faster than strongest visual baseline but lower F1. | results/image_text_fusion_308_nemotron_qwen3vl30b |
| 6 | Nemotron full image+text no reranker + Qwen30B | Full image+text | Nemotron image | None | full page + layout crop | Qwen3-VL-30B | 0.6784 | 0.7143 | 0.7029 | 0.6169 | 3.4263 | Fast full image+text ablation; text remains in VLM input, reranker removed. | results/image_text_full_308_nemotron_no_reranker_qwen3vl30b |
| 7 | Fusion ColPali + VL reranker + Qwen30B | Image+text fusion | ColPali/ColVision | Nemotron VL reranker | full page + layout crop | Qwen3-VL-30B | 0.6782 | 0.7143 | 0.6906 | 0.6474 | 11.8085 | Reference baseline. | results/image_text_fusion_308_colpali_qwen3vl30b |
| 8 | Fusion Nemotron no image reranker + Qwen30B | Image+text fusion | Nemotron image | None | full page + layout crop | Qwen3-VL-30B | 0.6575 | 0.6786 | 0.6786 | 0.6047 | 2.5080 | Very fast; fusion partly compensates for removing image reranker. | results/image_text_fusion_308_nemotron_no_reranker_qwen3vl30b |
| 9 | ColPali no reranker + Qwen30B | ColPali visual Qwen30B | ColPali/ColVision | None | full page + layout crop | Qwen3-VL-30B | 0.6539 | 0.6786 | 0.6718 | 0.6092 | 6.0709 | Speed ablation. | results/docbench_qwen3vl30b_no_reranker |
| 10 | Nemotron image no reranker + Qwen30B | Nemotron image retriever | Nemotron image | None | full page + layout crop | Qwen3-VL-30B | 0.6479 | 0.6688 | 0.6655 | 0.6040 | 2.5535 | Reference baseline. | results/multimodal_308_nemotron_image_retriever_qwen3vl30b |
| 11 | ColPali + VL reranker + Qwen8B | ColPali visual Qwen8B | ColPali/ColVision | Nemotron VL reranker | full page + layout crop | Qwen3-VL-8B | 0.6170 | 0.6201 | 0.6391 | 0.5617 | 15.2920 | Reference baseline. | results/multimodal_308_with_reranker |
| 12 | ColPali no reranker + Qwen8B | ColPali visual Qwen8B | ColPali/ColVision | None | full page + layout crop | Qwen3-VL-8B | 0.5657 | 0.5747 | 0.5911 | 0.5020 | 9.9253 | Speed ablation. | results/multimodal_308_no_reranker |
| 13 | BM25 + BGE-reranker-large + Qwen30B | BM25 text reranking | BM25 page_text | BGE-reranker-large | page_text | Qwen3-VL-30B | 0.5497 | 0.5714 | 0.5542 | 0.5385 | 1.2344 | Text-only baseline branch. | results/text_reranker_308_bge_large_qwen3vl30b |
| 14 | BM25 + BGE-reranker-base + Qwen30B | BM25 text reranking | BM25 page_text | BGE-reranker-base | page_text | Qwen3-VL-30B | 0.5429 | 0.5617 | 0.5477 | 0.5309 | 0.8689 | Text-only baseline branch. | results/text_reranker_308_bge_base_qwen3vl30b |
| 15 | BGE-large encoder + BGE-large reranker + Qwen30B | Text evidence encoder | BGE-large text encoder | BGE-reranker-large | OCR + page_text + captions + table_text | Qwen3-VL-30B | 0.5354 | 0.5390 | 0.5646 | 0.4625 | 1.5102 | Advisor text-evidence branch; useful ablation, not main method. | results/text_evidence_encoder_308_bge_large_en_v1_5_bge_reranker_large_qwen3vl30b |
| 16 | BM25 + Jina reranker + Qwen30B | BM25 text reranking | BM25 page_text | Jina reranker | page_text | Qwen3-VL-30B | 0.5303 | 0.5455 | 0.5501 | 0.4807 | 0.8927 | Text-only baseline branch. | results/text_reranker_308_jina_qwen3vl30b |
| 17 | BM25 + MiniLM reranker + Qwen30B | BM25 text reranking | BM25 page_text | MiniLM reranker | page_text | Qwen3-VL-30B | 0.5265 | 0.5390 | 0.5463 | 0.4770 | 0.7538 | Text-only baseline branch. | results/text_reranker_308_minilm_qwen3vl30b |
| 18 | BGE-base encoder + BGE-large reranker + Qwen30B | Text evidence encoder | BGE-base text encoder | BGE-reranker-large | OCR + page_text + captions + table_text | Qwen3-VL-30B | 0.5205 | 0.5390 | 0.5521 | 0.4414 | 1.4988 | Advisor text-evidence branch; useful ablation, not main method. | results/text_evidence_encoder_308_bge_base_en_v1_5_bge_reranker_large_qwen3vl30b |
| 19 | BM25 no reranker + Qwen30B | BM25 text reranking | BM25 page_text | None | page_text | Qwen3-VL-30B | 0.5144 | 0.5325 | 0.5269 | 0.4830 | 0.7146 | Text-only baseline branch. | results/text_reranker_308_no_reranker_qwen3vl30b |
| 20 | BGE-large encoder no reranker + Qwen30B | Text evidence encoder | BGE-large text encoder | None | OCR + page_text + captions + table_text | Qwen3-VL-30B | 0.4424 | 0.4545 | 0.4673 | 0.3801 | 0.9953 | Advisor text-evidence branch; useful ablation, not main method. | results/text_evidence_encoder_308_bge_large_en_v1_5_no_reranker_qwen3vl30b |

## Reranker Ablation

| Comparison | No reranker | With reranker | Mean F1 no | Mean F1 with | Delta F1 | Latency no | Latency with | Delta latency | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ColPali Qwen8B VL reranker | ColPali no reranker + Qwen8B | ColPali + VL reranker + Qwen8B | 0.5657 | 0.6170 | 0.0513 | 9.9253 | 15.2920 | 5.3667 | Quality improves, latency increases |
| ColPali Qwen30B VL reranker | ColPali no reranker + Qwen30B | ColPali + VL reranker + Qwen30B | 0.6539 | 0.6860 | 0.0321 | 6.0709 | 14.8791 | 8.8082 | Quality improves, latency increases |
| Nemotron image Qwen30B VL reranker | Nemotron image no reranker + Qwen30B | Nemotron image + VL reranker + Qwen30B | 0.6479 | 0.6902 | 0.0423 | 2.5535 | 11.0702 | 8.5167 | Quality improves, latency increases |
| Nemotron fusion Qwen30B image reranker | Fusion Nemotron no image reranker + Qwen30B | Fusion Nemotron + VL reranker + Qwen30B | 0.6575 | 0.6793 | 0.0219 | 2.5080 | 9.4122 | 6.9042 | Quality improves, latency increases |
| Nemotron full image+text Qwen30B text-image reranker | Nemotron full image+text no reranker + Qwen30B | Nemotron full image+text + VL reranker + Qwen30B | 0.6784 | 0.7023 | 0.0240 | 3.4263 | 13.6441 | 10.2179 | Quality improves, latency increases |
| BM25 text BGE-large reranker | BM25 no reranker + Qwen30B | BM25 + BGE-reranker-large + Qwen30B | 0.5144 | 0.5497 | 0.0353 | 0.7146 | 1.2344 | 0.5198 | Quality improves, latency increases |
| Text evidence BGE-large reranker | BGE-large encoder no reranker + Qwen30B | BGE-large encoder + BGE-large reranker + Qwen30B | 0.4424 | 0.5354 | 0.0930 | 0.9953 | 1.5102 | 0.5149 | Quality improves, latency increases |
