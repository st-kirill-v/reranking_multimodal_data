from src.core.metrics import vqa_metrics


def test_numeric_exact_normalizes_trailing_zeroes() -> None:
    metrics = vqa_metrics.compute_extended_metrics("The score is 2.230.", "2.23")

    assert metrics["numeric_exact_match"] == 1.0


def test_numeric_relaxed_tolerates_percent_small_diff() -> None:
    metrics = vqa_metrics.compute_extended_metrics("The rate was 28.65%.", "28.7%")

    assert metrics["numeric_relaxed_match"] == 1.0


def test_numeric_relaxed_requires_integer_exact() -> None:
    metrics = vqa_metrics.compute_extended_metrics("There were 153 stores.", "152 stores")

    assert metrics["numeric_relaxed_match"] == 0.0


def test_unit_match_normalizes_currency_and_scale() -> None:
    metrics = vqa_metrics.compute_extended_metrics("$5,490 million", "5,490 million USD")

    assert metrics["unit_match"] == 1.0


def test_entity_match_ignores_common_company_suffixes() -> None:
    metrics = vqa_metrics.compute_extended_metrics(
        "Toyota Master Trust Bank Ltd held the shares.",
        "Toyota Master Trust Bank",
    )

    assert metrics["entity_match"] == 1.0


def test_retrieval_hits_use_expected_folder_and_oracle_pages() -> None:
    row = {
        "question": "What is the value according to Table 2?",
        "expected_folder": "88",
        "oracle_pages": [{"page": 53}],
    }
    pages = [{"folder": "88", "page": 52}, {"folder": "88", "page": 53}]

    metrics = vqa_metrics.compute_retrieval_metrics(row, pages)

    assert metrics["doc_hit_at_k"] == 1.0
    assert metrics["page_hit_at_k"] == 1.0
    assert metrics["table_hit_at_k"] == 1.0


def test_summary_includes_f1_lt_0_5_subset() -> None:
    rows = [
        {
            "type": "multimodal-t",
            "exact": 0.0,
            "f1": 0.4,
            "numeric_exact_match": 1.0,
        },
        {
            "type": "multimodal-f",
            "exact": 1.0,
            "f1": 1.0,
            "numeric_exact_match": None,
        },
    ]

    summary = vqa_metrics.summarize_answer_results(rows)

    assert summary["metric_groups"]["overall"]["count"] == 2
    assert summary["metric_groups"]["f1_lt_0_5"]["count"] == 1
    assert summary["metric_groups"]["multimodal-t"]["numeric_exact_match"] == 1.0
