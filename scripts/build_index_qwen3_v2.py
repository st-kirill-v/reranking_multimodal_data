"""Compatibility entrypoint for the clean Qwen3 page-index builder.

Prefer:
    python scripts/build_page_index.py --index-name pages_qwen3
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_page_index import main


if __name__ == "__main__":
    main()
