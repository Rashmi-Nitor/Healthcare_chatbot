"""
scripts/ingest.py

CLI entry point for Person 1's ingestion pipeline.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --data-dir data/ --reset

Run this after adding or removing knowledge-base documents in `data/`.
Use --reset to ensure removed files are excluded from the rebuilt index.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import sys
from pathlib import Path

from app.core.constants import DATA_DIR, VECTOR_DB_DIR, FAISS_INDEX_NAME
from app.rag.rag_pipeline import ingest_knowledge_base
from app.rag.vector_store import reset_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the healthcare knowledge base into FAISS.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Folder containing PDF/DOCX/TXT files.")
    parser.add_argument("--index-name", type=str, default=FAISS_INDEX_NAME, help="FAISS index name.")
    parser.add_argument("--reset", action="store_true", help="Delete any existing index before ingesting.")
    args = parser.parse_args()

    if args.reset:
        logger.info("Resetting existing vector store at %s...", VECTOR_DB_DIR)
        reset_vector_store(VECTOR_DB_DIR)

    try:
        count = ingest_knowledge_base(data_dir=args.data_dir, index_name=args.index_name)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if count == 0:
        logger.warning("No chunks were indexed. Check that %s contains supported files.", args.data_dir)
        return 1

    logger.info("Done. Indexed %d chunk(s) into '%s'.", count, args.index_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
