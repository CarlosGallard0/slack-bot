import os
import logging
from typing import List
from langchain_core.documents import Document
import time

from raptor_pipeline import RaptorBuilder
from raptor_ingestion.extractor_text import extract_from_pdf
from raptor_ingestion.section_selector import extract_section_context
from raptor_pipeline.gcs import GCSCheckpointStore


PDF_FOLDER = "docs"
RAPTOR_COLLECTION = "medical_papers"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_docs_for_raptor(force_rebuild: bool = False):
    start_time = time.perf_counter()

    documents: List[Document] = []

    logger.info(f"📂 Starting ingestion from folder: {PDF_FOLDER}")
    t_extract_start = time.perf_counter()
    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        logger.info(f"📄 Processing PDF: {filename}")
        pdf_path = os.path.join(PDF_FOLDER, filename)

        result = extract_from_pdf(pdf_path)

        if result.get("status") != "ok":
            logger.warning(
                f"⚠️ Skipped {filename} | status={result.get('status')} | reason={result.get('reason')}"
            )
            continue

        section, context = extract_section_context(
            full_text=result.get("full_text"),
            first_page_text=result.get("first_page_text"),
        )

        if not section or not context:
            logger.warning(f"⚠️ No valid section/context found for {filename}")
            continue

        logger.info(
            f"✅ Extracted section '{section}' ({len(context)} chars) from {filename}"
        )

        documents.append(
            Document(
                page_content=context,
                metadata={
                    "source": result["filename"],
                    "section_type": section,
                    "doi": result.get("doi"),
                    "year": result.get("year"),
                },
            )
        )
    logger.info(f"⏱️ PDF extraction took {time.perf_counter() - t_extract_start:.2f}s")
    logger.info(f"📦 Total documents prepared for RAPTOR: {len(documents)}")

    if not documents:
        raise RuntimeError("No valid documents for RAPTOR ingestion")

    # GCS CHECKPOINTS
    BUCKET_NAME = "raptor-checkpoints-neurodocsdomain"
    COLLECTION_NAME = RAPTOR_COLLECTION

    LOCAL_CHECKPOINT_DIR = f".raptor_checkpoints/{COLLECTION_NAME}"
    GCS_PREFIX = COLLECTION_NAME

    checkpoint_store = GCSCheckpointStore(
        bucket_name=BUCKET_NAME,
        prefix=GCS_PREFIX,
        local_dir=LOCAL_CHECKPOINT_DIR,
    )

    if os.path.exists(LOCAL_CHECKPOINT_DIR):
        import shutil

        shutil.rmtree(LOCAL_CHECKPOINT_DIR)
    checkpoint_store.pull()

    logger.info("🌳 Initializing RAPTOR...")

    """raptor = RaptorLangChain(
        collection_name=RAPTOR_COLLECTION,
        max_depth=3,
        chunk_size=800,
        chunk_overlap=100,
        retrieval_mode="collapsed",
    )"""

    raptor = RaptorBuilder(
        collection_name=RAPTOR_COLLECTION,
        max_depth=3,
        chunk_size=800,
        chunk_overlap=100,
    )

    logger.info("🚀 Building RAPTOR tree (this may take several minutes)...")
    t_build_start = time.perf_counter()
    raptor.build_tree(documents, force_rebuild=force_rebuild)
    checkpoint_store.push()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    logger.info(f"⏱️ RAPTOR build took {time.perf_counter() - t_build_start:.2f}s")

    logger.info(
        f"⏱️ RAPTOR ingestion finished in {elapsed:.2f} seconds "
        f"({elapsed/60:.2f} minutes)"
    )

    logger.info("✅ RAPTOR ingestion completed successfully")


if __name__ == "__main__":
    ingest_docs_for_raptor(force_rebuild=False)
