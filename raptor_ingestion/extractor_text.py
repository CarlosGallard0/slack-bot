import re
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

from raptor_ingestion.identifiers_from_docs import (
    extract_doi,
    extract_publication_year,
)

EXCLUDED_DOC_PATTERNS = [
    r"permissions@lww.com",
    r"Lippincott Journal Portfolio Author Permission Guidelines",
]

COMPILED_EXCLUDED = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_DOC_PATTERNS]


def extract_year_from_metadata(metadata: dict) -> int | None:
    creation_date = metadata.get("creationdate")
    if isinstance(creation_date, str):
        match = re.search(r"(19\d{2}|20\d{2})", creation_date)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= datetime.now().year:
                return year
    return None


def extract_from_pdf(pdf_path: str) -> dict:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        return {
            "status": "skip",
            "reason": "empty_pdf",
            "filename": Path(pdf_path).name,
        }

    first_page = pages[0]
    first_page_text = first_page.page_content
    metadata = first_page.metadata

    for rx in COMPILED_EXCLUDED:
        if rx.search(first_page_text):
            return {
                "status": "excluded",
                "reason": "excluded_doc_pattern",
                "filename": Path(pdf_path).name,
            }

    doi = extract_doi(first_page_text)
    year = extract_year_from_metadata(metadata) or extract_publication_year(
        first_page_text
    )

    full_text = "\n\n".join(p.page_content for p in pages)

    return {
        "status": "ok",
        "filename": Path(pdf_path).name,
        "first_page_text": first_page_text,
        "full_text": full_text,
        "metadata": metadata,
        "doi": doi,
        "year": year,
    }
