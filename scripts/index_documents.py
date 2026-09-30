"""
SafetyCopilot — Real HSE Document Indexer

Pipeline:
PDFs -> Page Extraction -> Chunking -> Embeddings -> Qdrant

Indexes all HSE document chunks into the persistent
SafetyCopilot Qdrant collection.
"""

from pathlib import Path
import sys


# ---------------------------------------------------------
# Project Setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.qdrant_store import QdrantVectorStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "safetycopilot"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 32


def main():

    print("=" * 100)
    print("  SAFETYCOPILOT — HSE DOCUMENT INDEXING")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load PDFs
    # -----------------------------------------------------

    print(f"\n[1/5] Loading PDFs from:")
    print(f"      {RAW_DATA_DIR}")

    documents = load_pdfs_from_directory(
        RAW_DATA_DIR,
        recursive=True,
    )

    if not documents:
        print("\n[ERROR] No PDF documents were loaded.")
        return

    total_pages = sum(
        document.total_pages
        for document in documents
    )

    print(f"\n      Documents : {len(documents)}")
    print(f"      Pages     : {total_pages}")

    # -----------------------------------------------------
    # 2. Chunk Documents
    # -----------------------------------------------------

    print("\n[2/5] Creating text chunks...")

    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        print("\n[ERROR] No chunks were generated.")
        return

    print(f"      Chunks created : {len(chunks)}")

    # -----------------------------------------------------
    # 3. Load Embedding Model
    # -----------------------------------------------------

    print("\n[3/5] Loading embedding model...")

    embedding_model = EmbeddingModel()

    print(
        f"      Model dimension : "
        f"{embedding_model.dimension}"
    )

    # -----------------------------------------------------
    # 4. Generate Embeddings
    # -----------------------------------------------------

    print("\n[4/5] Generating embeddings...")

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_texts(
        texts=texts,
        batch_size=BATCH_SIZE,
    )

    print(
        f"\n      Embedding matrix : "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------
    # Prepare Qdrant payloads
    # -----------------------------------------------------

    ids = list(
        range(1, len(chunks) + 1)
    )

    payloads = []

    for chunk in chunks:

        payloads.append(
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "filename": chunk.filename,
                "source_path": chunk.source_path,
                "category": chunk.category,
                "page_number": chunk.page_number,
                "total_pages": chunk.total_pages,
            }
        )

    # -----------------------------------------------------
    # 5. Store in Qdrant
    # -----------------------------------------------------

    print("\n[5/5] Indexing vectors into Qdrant...")

    store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        vector_size=embedding_model.dimension,
        storage_path=QDRANT_PATH,
    )

    try:

        # Recreate collection so repeated indexing
        # does not create stale/duplicate data.
        store.create_collection(
            recreate=True
        )

        store.add_points(
            ids=ids,
            vectors=embeddings.tolist(),
            payloads=payloads,
        )

        stored_count = store.count()

        print(
            f"\n      Points stored : "
            f"{stored_count}"
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        print("\n" + "=" * 100)
        print("  INDEXING RESULT")
        print("=" * 100)

        if stored_count == len(chunks):

            print(
                "\n  [PASS] All HSE chunks were "
                "successfully indexed."
            )

            print(
                f"  Documents : {len(documents)}"
            )

            print(
                f"  Pages     : {total_pages}"
            )

            print(
                f"  Chunks    : {len(chunks)}"
            )

            print(
                f"  Vectors   : {stored_count}"
            )

            print(
                f"  Dimension : "
                f"{embedding_model.dimension}"
            )

            print(
                f"  Collection: {COLLECTION_NAME}"
            )

        else:

            print(
                "\n  [WARNING] Stored vector count "
                "does not match generated chunks."
            )

            print(
                f"  Expected : {len(chunks)}"
            )

            print(
                f"  Stored   : {stored_count}"
            )

    finally:

        store.close()

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()