from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mmrag.config import ProjectPaths
from src.mmrag.dataset import load_docbench_questions
from src.mmrag.oracle import find_answer_pages, load_pages_text


def safe_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DocBench QA-folder and answer oracle alignment."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--types", nargs="*", default=["multimodal-t", "multimodal-f"])
    parser.add_argument("--output", type=Path, default=Path("data/audit_docbench_alignment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(data_dir=args.data_dir)
    questions = load_docbench_questions(paths.data_dir, question_types=set(args.types))
    if args.limit > 0:
        questions = questions[: args.limit]

    rows = []
    answer_found = 0
    exact_found = 0
    for i, q in enumerate(questions, start=1):
        folder_path = paths.data_dir / str(q["folder"])
        pages_text = load_pages_text(folder_path)
        matches = find_answer_pages(folder_path, q["answer"], q.get("evidence", ""))
        if matches:
            answer_found += 1
        if any(match.exact_answer for match in matches):
            exact_found += 1
        row = {
            "folder": str(q["folder"]),
            "question": q["question"],
            "answer": q["answer"],
            "type": q["type"],
            "num_pages_text": len(pages_text),
            "oracle_pages": [
                {
                    "page": match.page,
                    "exact_answer": match.exact_answer,
                    "number_recall": match.number_recall,
                    "keyword_recall": match.keyword_recall,
                    "matched_numbers": match.matched_numbers,
                    "matched_keywords": match.matched_keywords,
                }
                for match in matches[:10]
            ],
        }
        rows.append(row)
        best = row["oracle_pages"][0] if row["oracle_pages"] else None
        message = (
            f"[{i}/{len(questions)}] folder={row['folder']} pages={row['num_pages_text']} "
            f"oracle_page={best['page'] if best else None} "
            f"num_recall={best['number_recall'] if best else 0:.2f} "
            f"kw_recall={best['keyword_recall'] if best else 0:.2f} "
            f"question={q['question'][:90]}"
        )
        print(safe_console(message))

    summary = {
        "total": len(rows),
        "answer_oracle_found_rate": answer_found / max(len(rows), 1),
        "exact_answer_found_rate": exact_found / max(len(rows), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
