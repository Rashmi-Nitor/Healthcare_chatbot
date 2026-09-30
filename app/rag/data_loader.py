"""
app/rag/data_loader.py

Phase 1 — Document ingestion.

Responsible for:
- Loading PDF / DOCX / TXT files from a directory (or a single file).
- Attaching consistent metadata (source filename, doc type, ingestion time,
  page number where available) to every LangChain Document.
- Basic validation (supported extension, non-empty content).

This module has no knowledge of chunking, embeddings, or the vector store —
it only produces a list of raw `langchain_core.documents.Document` objects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from app.core.constants import SUPPORTED_EXTENSIONS, DATA_DIR

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension is not one we know how to ingest."""


class EmptyDocumentError(Exception):
    """Raised when a loaded file produced no usable text content."""


def _ocr_pdf(path: Path) -> List[Document]:
    """Extract text from a scanned PDF one page at a time using OCR."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are required for scanned PDFs. "
            "Install pymupdf, pytesseract, and Pillow."
        ) from exc

    documents: List[Document] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = pytesseract.image_to_string(image).strip()
            if text:
                documents.append(Document(page_content=text, metadata={"page": page_number}))
    return documents


def _loader_for(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    raise UnsupportedFileTypeError(
        f"'{ext}' is not supported. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def load_file(path: str | Path) -> List[Document]:
    """
    Load a single file and return a list of LangChain Documents (one per
    page for PDFs, one document for DOCX/TXT), each tagged with metadata.

    Raises:
        FileNotFoundError, UnsupportedFileTypeError, EmptyDocumentError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{path.suffix}' is not supported. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Loading document: %s", path.name)
    loader = _loader_for(path)

    try:
        raw_docs = loader.load()
        if path.suffix.lower() == ".pdf" and all(
            not doc.page_content.strip() for doc in raw_docs
        ):
            logger.info("No text layer found in %s; trying OCR.", path.name)
            raw_docs = _ocr_pdf(path)
    except Exception as exc:  # noqa: BLE001 - we want to wrap & log any loader failure
        logger.exception("Failed to load %s", path.name)
        raise RuntimeError(f"Failed to load '{path.name}': {exc}") from exc

    if not raw_docs or all(not d.page_content.strip() for d in raw_docs):
        raise EmptyDocumentError(f"'{path.name}' produced no extractable text.")

    ingested_at = datetime.now(timezone.utc).isoformat()
    enriched: List[Document] = []
    for i, doc in enumerate(raw_docs):
        if not doc.page_content.strip():
            continue  # skip blank pages
        doc.metadata.update(
            {
                "source": path.name,
                "source_path": str(path),
                "doc_type": path.suffix.lower().lstrip("."),
                "page": doc.metadata.get("page", i),
                "ingested_at": ingested_at,
            }
        )
        enriched.append(doc)

    logger.info("Loaded %d content unit(s) from %s", len(enriched), path.name)
    return enriched


def load_directory(directory: str | Path = DATA_DIR) -> List[Document]:
    """
    Walk `directory` and load every supported file inside it.
    Files that fail to load are logged and skipped rather than aborting the
    whole ingestion run.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_docs: List[Document] = []
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", directory)
        return all_docs

    for path in files:
        try:
            all_docs.extend(load_file(path))
        except (UnsupportedFileTypeError, EmptyDocumentError, RuntimeError) as exc:
            logger.error("Skipping '%s': %s", path.name, exc)
            continue

    logger.info("Ingestion complete: %d document chunks loaded from %d file(s).",
                len(all_docs), len(files))
    return all_docs
