from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.metadata_tools import (  # noqa: E402
    detect_mention_count_phrase,
    detect_page_count,
    detect_page_reference,
    normalize_text_route_answer,
    run_metadata_tool,
)


def main() -> None:
    assert detect_page_count("How many pages does the document have?")
    assert detect_page_reference("What is the main content of page 10?") == 10
    assert detect_mention_count_phrase('How many times does the report mention "PM"?') == "PM"
    assert (
        detect_mention_count_phrase("How many time does the paper mention WikiText-2?")
        == "WikiText-2"
    )
    assert normalize_text_route_answer("not mentioned in the context") == "NOT FOUND"
    assert normalize_text_route_answer("Yes, the text says so.") == "Yes."
    assert normalize_text_route_answer("No, the text does not mention it.") == "No."

    tool = run_metadata_tool(
        question="How many pages does the document have?",
        question_type="meta-data",
        data_dir=Path("data/datasets/docbench"),
        doc_id="0",
        context_max_chars=12000,
    )
    if tool is not None:
        assert tool.tool_used == "page_count"
        assert tool.answer is not None

    direct = run_metadata_tool(
        question="What is the main content of page 5?",
        question_type="meta-data",
        data_dir=Path("data/datasets/docbench"),
        doc_id="0",
        context_max_chars=12000,
    )
    if direct is not None:
        assert direct.tool_used == "page_direct"
        assert direct.pages and direct.pages[0].page == 5

    print("Text route tools smoke test passed.")


if __name__ == "__main__":
    main()
