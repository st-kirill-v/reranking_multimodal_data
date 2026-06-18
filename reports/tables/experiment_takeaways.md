# Experiment Takeaways

Source table: `reports/tables/paper_multimodal_308.csv`.

## Headline

- Best overall: `image_text_full_308_nemotron_qwen3vl30b` with mean_f1=0.7023 and latency_mean=13.6441s.
- Previous best image+text input: `image_text_input_308_nemotron_qwen3vl30b` with mean_f1=0.6979 and latency_mean=12.9920s.
- Full text+image gain over previous image+text input: mean_f1 +0.0045; latency +0.6521s.
- Full text+image gain over image-only best: mean_f1 +0.0121; latency +2.5739s.
- Full text+image reranker effect: mean_f1 +0.0240 over the no-reranker full image+text ablation; latency +10.2179s.
- Best text-only/text-evidence branch: `text_reranker_308_bge_large_qwen3vl30b` with mean_f1=0.5497.
- Best score-fusion branch: `image_text_fusion_308_nemotron_qwen3vl30b` with mean_f1=0.6793 and latency_mean=9.4122s.
- Fast visual speed ablation: `image_text_fusion_308_nemotron_no_reranker_qwen3vl30b` with latency_mean=2.5080s and mean_f1=0.6575.

## Interpretation

- The new strongest result is the full image+text pipeline: Nemotron image retrieval, Nemotron text+image VL reranking, layout-aware page/crop images, and selected-page text evidence injected into Qwen3-VL-30B.
- This is the closest implemented experiment to the advisor text+image idea because text is used in both reranking and final generation.
- The matching no-reranker ablation keeps the same Nemotron image retrieval and the same selected-page text evidence in Qwen3-VL-30B, but removes the text+image VL reranker.
- Removing the reranker reduces latency from 13.64s to 3.43s, but drops mean_f1 from 0.7023 to 0.6784 and f1>0.5 from 0.7565 to 0.7143.
- Compared with the previous image+text input run, adding text to the Nemotron reranker improves mean_f1 from 0.6979 to 0.7023.
- Compared with the image-only Nemotron reranker baseline, the full text+image pipeline improves mean_f1 from 0.6902 to 0.7023.
- The gain is strongest on multimodal-t: 0.7201 versus 0.7015 for the image-only baseline.
- Multimodal-f is roughly on par with the best visual baselines: 0.6578 versus 0.6620 for image-only and 0.6579 for VLM text-input only.
- Latency increases: 13.64s versus 12.99s for text only in VLM input and 11.07s for image-only reranking/generation.

## Main Result Comparison

| Method | Mean F1 | F1>0.5 | Exact | MM-T F1 | MM-F F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| `image_text_full_308_nemotron_qwen3vl30b` | 0.7023 | 0.7565 | 0.1201 | 0.7201 | 0.6578 | 13.6441s |
| `image_text_input_308_nemotron_qwen3vl30b` | 0.6979 | 0.7468 | 0.1169 | 0.7138 | 0.6579 | 12.9920s |
| `multimodal_308_nemotron_image_retriever_with_reranker_qwen3vl30b` | 0.6902 | 0.7500 | 0.1071 | 0.7015 | 0.6620 | 11.0702s |
| `image_text_fusion_308_nemotron_qwen3vl30b` | 0.6793 | 0.7175 | 0.1136 | 0.6933 | 0.6445 | 9.4122s |
| `image_text_full_308_nemotron_no_reranker_qwen3vl30b` | 0.6784 | 0.7143 | 0.1201 | 0.7029 | 0.6169 | 3.4263s |
| `image_text_fusion_308_nemotron_no_reranker_qwen3vl30b` | 0.6575 | 0.6786 | 0.1136 | 0.6786 | 0.6047 | 2.5080s |
| `text_reranker_308_bge_large_qwen3vl30b` | 0.5497 | 0.5714 | 0.0812 | 0.5542 | 0.5385 | 1.2344s |

## Reranker Effect

| Comparison | Mean F1 no | Mean F1 with | Delta F1 | Latency no | Latency with | Delta latency |
|---|---:|---:|---:|---:|---:|---:|
| ColPali/ColVision + Qwen8B: VL reranker effect | 0.5657 | 0.6170 | +0.0513 | 9.9253 | 15.2920 | +5.3667 |
| ColPali/ColVision + Qwen30B: VL reranker effect | 0.6539 | 0.6860 | +0.0321 | 6.0709 | 14.8791 | +8.8082 |
| Nemotron image retriever + Qwen30B: VL reranker effect | 0.6479 | 0.6902 | +0.0423 | 2.5535 | 11.0702 | +8.5167 |
| Nemotron image+text fusion + Qwen30B: image reranker effect | 0.6575 | 0.6793 | +0.0219 | 2.5080 | 9.4122 | +6.9042 |
| Nemotron full image+text + Qwen30B: text+image reranker effect | 0.6784 | 0.7023 | +0.0240 | 3.4263 | 13.6441 | +10.2179 |
| BM25 page_text + Qwen30B: BGE-large text reranker effect | 0.5144 | 0.5497 | +0.0353 | 0.7146 | 1.2344 | +0.5198 |
| Text evidence encoder + Qwen30B: BGE-large text reranker effect | 0.4424 | 0.5354 | +0.0930 | 0.9953 | 1.5102 | +0.5149 |

## Recommended Paper Framing

- Main method: Nemotron image retriever + Nemotron text+image VL reranker + layout-aware images/crops + selected-page text evidence in Qwen3-VL-30B input.
- Key baseline: same visual pipeline without text in reranker/VLM input.
- Strong text baseline: BM25 page_text + BGE-reranker-large + Qwen3-VL-30B.
- Fusion ablation: score-level image+text fusion demonstrates the advisor formula with retrieval, text rerank, image rerank, and lexical/table boosts.
