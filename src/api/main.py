"""
SafetyCopilot — FastAPI Backend

Provides:
- Health monitoring
- Dynamic HSE document management
- PDF source serving
- Hybrid RAG question answering
- Conversation memory
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.generation.rag_generator import RAGGenerator
from src.knowledge_base.knowledge_base_manager import (
    KnowledgeBaseManager,
)
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.reranker import SafetyReranker


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

QDRANT_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIGURATION
# =========================================================

COLLECTION_NAME = "safetycopilot"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DENSE_TOP_K = 15
RRF_K = 60
EMBEDDING_BATCH_SIZE = 32

MAX_UPLOAD_SIZE = 25 * 1024 * 1024

# Prevent two upload/delete rebuilds from running
# at the same time.
document_management_lock = threading.RLock()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="SafetyCopilot API",
    description=(
        "RAG-based Industrial HSE Knowledge Assistant"
    ),
    version="1.1.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# API MODELS
# =========================================================

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class SourceItem(BaseModel):
    filename: str
    page: int
    category: str


class DocumentItem(BaseModel):
    filename: str
    category: str


class AskResponse(BaseModel):
    question: str
    session_id: str
    answer: str
    sources: list[SourceItem]
    provider: str | None = None
    model: str | None = None
    grounded: bool


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    category: str
    documents: int
    chunks: int
    vectors: int


class DocumentDeleteResponse(BaseModel):
    message: str
    filename: str
    documents: int
    chunks: int
    vectors: int


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_category_name(
    category: str | None,
) -> str:
    """
    Convert a user-supplied category into a safe directory
    name.

    Example:
        "Fire Safety" -> "fire_safety"
    """

    if not category:
        return "uploaded"

    normalized = (
        category.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    # Keep only safe directory-name characters.
    normalized = "".join(
        character
        for character in normalized
        if character.isalnum()
        or character == "_"
    )

    normalized = normalized.strip("_")

    return normalized or "uploaded"


def display_category(
    pdf_path: Path,
) -> str:
    """
    Determine the display category of a PDF using its
    folder inside data/raw.
    """

    try:
        relative_path = pdf_path.relative_to(
            RAW_DATA_DIR
        )

        if len(relative_path.parts) > 1:
            return (
                relative_path.parent.name
                .replace("_", " ")
                .title()
            )

    except ValueError:
        pass

    return "HSE Document"


def find_document(
    filename: str,
) -> Path | None:
    """
    Find one PDF recursively by safe filename.
    """

    safe_filename = Path(filename).name

    if not safe_filename:
        return None

    for pdf_path in RAW_DATA_DIR.rglob("*.pdf"):
        if (
            pdf_path.name.lower()
            == safe_filename.lower()
        ):
            return pdf_path

    return None


def document_exists(
    filename: str,
) -> bool:
    """
    Check whether a PDF with this filename already exists
    anywhere in the HSE knowledge base.
    """

    return find_document(filename) is not None


def remove_empty_parent_directories(
    start_directory: Path,
) -> None:
    """
    Remove empty category directories after a document is
    deleted, but never remove data/raw itself.
    """

    current = start_directory

    while (
        current != RAW_DATA_DIR
        and RAW_DATA_DIR in current.parents
    ):
        try:
            current.rmdir()
        except OSError:
            break

        current = current.parent


# =========================================================
# INITIALIZE RAG SYSTEM
# =========================================================

print(
    "Initializing SafetyCopilot RAG pipeline..."
)


knowledge_base = KnowledgeBaseManager(
    raw_data_dir=RAW_DATA_DIR,
    qdrant_path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    dense_top_k=DENSE_TOP_K,
    rrf_k=RRF_K,
    embedding_batch_size=EMBEDDING_BATCH_SIZE,
)


try:
    initial_stats = knowledge_base.initialize(
        rebuild_vectors=False
    )

except Exception as exc:
    raise RuntimeError(
        "Unable to initialize SafetyCopilot "
        f"knowledge base: {exc}"
    ) from exc


reranker = SafetyReranker()


generator = RAGGenerator(
    gemini_api_key="",
    gemini_retries=1,
    retry_delay=1,
)


memory = ConversationMemory(
    max_turns=5,
)


print(
    "SafetyCopilot ready: "
    f"{initial_stats['documents']} documents, "
    f"{initial_stats['chunks']} chunks, "
    f"{initial_stats['vectors']} vectors."
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "service": "SafetyCopilot API",
        "status": "running",
        "version": "1.1.0",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    stats = knowledge_base.stats()

    return {
        "status": "healthy",
        "service": "SafetyCopilot",
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "vectors": stats["vectors"],
    }


# =========================================================
# LIST DOCUMENTS
# =========================================================

@app.get(
    "/documents",
    response_model=list[DocumentItem],
)
def get_documents():
    """
    Return all PDF documents currently available in the
    HSE knowledge base.
    """

    items: list[DocumentItem] = []

    for pdf_path in sorted(
        RAW_DATA_DIR.rglob("*.pdf"),
        key=lambda path: (
            path.name.lower()
        ),
    ):
        try:
            items.append(
                DocumentItem(
                    filename=pdf_path.name,
                    category=display_category(
                        pdf_path
                    ),
                )
            )

        except Exception:
            continue

    return items


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
)
def upload_document(
    file: UploadFile = File(...),
    category: str = Form("uploaded"),
):
    """
    Upload a new HSE PDF and immediately rebuild the
    retrieval knowledge base.

    The new PDF becomes available to RAG without requiring
    a backend restart.
    """

    original_filename = (
        Path(file.filename or "").name
    )

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if (
        Path(original_filename)
        .suffix
        .lower()
        != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF documents can be uploaded."
            ),
        )

    safe_category = normalize_category_name(
        category
    )

    with document_management_lock:

        if document_exists(
            original_filename
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A document with this filename "
                    "already exists in the knowledge base."
                ),
            )

        target_directory = (
            RAW_DATA_DIR
            / safe_category
        )

        target_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            target_directory
            / original_filename
        )

        file_created = False

        try:
            # ---------------------------------------------
            # Save PDF
            # ---------------------------------------------

            file.file.seek(0)

            with target_path.open(
                "wb"
            ) as output_file:
                shutil.copyfileobj(
                    file.file,
                    output_file,
                )

            file_created = True

            # ---------------------------------------------
            # Basic file validation
            # ---------------------------------------------

            if not target_path.exists():
                raise ValueError(
                    "Uploaded file could not be saved."
                )

            file_size = (
                target_path.stat().st_size
            )

            if file_size == 0:
                raise ValueError(
                    "Uploaded PDF is empty."
                )

            if file_size > MAX_UPLOAD_SIZE:
                raise ValueError(
                    "Uploaded PDF exceeds the "
                    "25 MB size limit."
                )

            # Validate PDF signature.
            with target_path.open(
                "rb"
            ) as pdf_file:
                signature = (
                    pdf_file.read(5)
                )

            if signature != b"%PDF-":
                raise ValueError(
                    "The uploaded file is not "
                    "a valid PDF document."
                )

            # ---------------------------------------------
            # Rebuild complete knowledge base
            # ---------------------------------------------

            # ---------------------------------------------
            # Incrementally index uploaded document
            # ---------------------------------------------

            stats = (
                knowledge_base.add_document(
                    original_filename
                )
            )

            # Make sure the document loader actually
            # accepted the uploaded document.
            loaded_filenames = {
                document.filename.lower()
                for document
                in knowledge_base.documents
            }

            if (
                original_filename.lower()
                not in loaded_filenames
            ):
                raise ValueError(
                    "The PDF could not be processed "
                    "into the HSE knowledge base."
                )

            return DocumentUploadResponse(
                message=(
                    "Document uploaded and indexed "
                    "successfully."
                ),
                filename=original_filename,
                category=(
                    safe_category
                    .replace("_", " ")
                    .title()
                ),
                documents=stats[
                    "documents"
                ],
                chunks=stats["chunks"],
                vectors=stats["vectors"],
            )

        except HTTPException:
            raise

        except Exception as exc:

            # ---------------------------------------------
            # Rollback failed upload
            # ---------------------------------------------

            if (
                file_created
                and target_path.exists()
            ):
                try:
                    target_path.unlink()
                except Exception:
                    pass

            remove_empty_parent_directories(
                target_directory
            )

            # Restore retrieval state from the PDFs that
            # existed before this failed upload.
            

            raise HTTPException(
                status_code=400,
                detail=(
                    "Document upload failed: "
                    f"{str(exc)}"
                ),
            ) from exc

        finally:
            try:
                file.file.close()
            except Exception:
                pass


# =========================================================
# DELETE DOCUMENT
# =========================================================


@app.delete(
    "/documents/{filename}",
    response_model=DocumentDeleteResponse,
)
def delete_document(
    filename: str,
):
    """
    Delete one HSE PDF from the SafetyCopilot knowledge base.

    The KnowledgeBaseManager performs document-specific
    vector deletion and refreshes the in-memory retrieval
    state. The complete Qdrant collection is not rebuilt.
    """

    safe_filename = Path(filename).name

    if not safe_filename:
        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty.",
        )

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents can be deleted.",
        )

    with document_management_lock:

        pdf_path = find_document(
            safe_filename
        )

        if (
            pdf_path is None
            or not pdf_path.is_file()
            or pdf_path.suffix.lower() != ".pdf"
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Document not found in the "
                    "knowledge base."
                ),
            )

        parent_directory = pdf_path.parent

        try:
            result = (
                knowledge_base.delete_document(
                    safe_filename
                )
            )

            remove_empty_parent_directories(
                parent_directory
            )

            return DocumentDeleteResponse(
                message=(
                    "Document deleted from the "
                    "knowledge base successfully."
                ),
                filename=safe_filename,
                documents=int(
                    result["documents"]
                ),
                chunks=int(
                    result["chunks"]
                ),
                vectors=int(
                    result["vectors"]
                ),
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Document deletion failed: "
                    f"{str(exc)}"
                ),
            ) from exc

# =========================================================
# SERVE SOURCE PDF
# =========================================================

@app.get("/sources/{filename}")
def get_source_document(
    filename: str,
):
    """
    Serve an HSE source PDF from the knowledge base.
    """

    safe_filename = Path(
        filename
    ).name

    pdf_path = find_document(
        safe_filename
    )

    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Source document not found."
            ),
        )

    if (
        not pdf_path.is_file()
        or pdf_path.suffix.lower()
        != ".pdf"
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Source document not found."
            ),
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=safe_filename,
        content_disposition_type="inline",
    )


# =========================================================
# ASK SAFETYCOPILOT
# =========================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_safetycopilot(
    request: AskRequest,
):
    """
    Ask SafetyCopilot an HSE question.

    Conversation context is used to resolve follow-up
    references during retrieval. Generation receives the
    original user question plus retrieved evidence.
    """

    question = (
        request.question.strip()
    )

    session_id = (
        request.session_id.strip()
        or "default"
    )

    if not question:
        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot be empty."
            ),
        )

    try:
        # ---------------------------------------------
        # Empty Knowledge Base Guard
        # ---------------------------------------------

        if (
            knowledge_base.document_count
            == 0
            or knowledge_base.chunk_count
            == 0
        ):
            return AskResponse(
                question=question,
                session_id=session_id,
                answer=(
                    "The HSE knowledge base is empty. "
                    "Upload an HSE PDF document before "
                    "asking knowledge-base questions."
                ),
                sources=[],
                provider=None,
                model=None,
                grounded=False,
            )

        # ---------------------------------------------
        # Conversation Context
        # ---------------------------------------------

        retrieval_query = (
            memory.build_context(
                session_id=session_id,
                current_question=(
                    question
                ),
                history_turns=2,
            )
        )

        print(
            "\n--- CONVERSATION CONTEXT ---"
        )

        print(
            f"Session: {session_id}"
        )

        print(
            retrieval_query
        )

        print(
            "----------------------------\n"
        )

        # ---------------------------------------------
        # Hybrid Retrieval
        # ---------------------------------------------

        hybrid_candidates = (
            knowledge_base.retrieve(
                query=retrieval_query,
                top_k=10,
                candidate_k=15,
            )
        )

        if not hybrid_candidates:
            return AskResponse(
                question=question,
                session_id=session_id,
                answer=(
                    "The available HSE documents do not "
                    "provide enough relevant information "
                    "to answer this question."
                ),
                sources=[],
                provider=None,
                model=None,
                grounded=False,
            )

        # ---------------------------------------------
        # Reranking
        # ---------------------------------------------

        evidence = reranker.rerank(
            query=retrieval_query,
            candidates=hybrid_candidates,
            top_k=3,
        )

        print(
            "\n--- RERANKER DEBUG ---"
        )

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            score = float(
                item.get(
                    "reranker_score",
                    0.0,
                )
            )

            print(
                f"{index}. "
                f"{item.get('filename', '')} | "
                f"score={score:.4f}"
            )

            print(
                "TEXT: "
                f"{item.get('text', '')[:1000]}"
            )

            print()

        print(
            "----------------------\n"
        )

        # ---------------------------------------------
        # Relevance Guard
        # ---------------------------------------------
        #
        # Preserve the threshold used by the current
        # working/evaluated SafetyCopilot pipeline.
        #
        # Conversation follow-ups can receive slightly
        # negative cross-encoder scores even when the
        # correct evidence has been retrieved.
        # ---------------------------------------------

        RELEVANCE_THRESHOLD = -1.0

        relevant_evidence = [
            item
            for item in evidence
            if float(
                item.get(
                    "reranker_score",
                    -999.0,
                )
            )
            >= RELEVANCE_THRESHOLD
        ]

        if not relevant_evidence:
            return AskResponse(
                question=question,
                session_id=session_id,
                answer=(
                    "The available HSE documents do not "
                    "provide enough relevant information "
                    "to answer this question."
                ),
                sources=[],
                provider=None,
                model=None,
                grounded=False,
            )

        evidence = relevant_evidence

        # ---------------------------------------------
        # Generation
        # ---------------------------------------------

        result = generator.generate(
            question=question,
            evidence=evidence,
        )

        # ---------------------------------------------
        # Source Validation
        # ---------------------------------------------

        sources: list[
            SourceItem
        ] = []

        for source in result.get(
            "sources",
            [],
        ):
            filename = (
                source.get(
                    "filename",
                    "",
                )
            )

            page_number = (
                source.get(
                    "page_number"
                )
            )

            category = (
                source.get(
                    "category",
                    "",
                )
            )

            if (
                not filename
                or page_number is None
            ):
                continue

            try:
                page_number = int(
                    page_number
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            sources.append(
                SourceItem(
                    filename=filename,
                    page=page_number,
                    category=category,
                )
            )

        grounded = bool(
            result.get(
                "grounded",
                False,
            )
        )

        # A grounded answer should have traceable source
        # metadata. If generation marks the answer as
        # grounded without any valid source, fail closed.
        if grounded and not sources:
            grounded = False

        # ---------------------------------------------
        # Conversation Memory
        # ---------------------------------------------

        if grounded:
            memory.add_turn(
                session_id=session_id,
                question=question,
                answer=result["answer"],
            )

        return AskResponse(
            question=question,
            session_id=session_id,
            answer=result["answer"],
            sources=(
                sources
                if grounded
                else []
            ),
            provider=result.get(
                "provider"
            ),
            model=result.get(
                "model"
            ),
            grounded=grounded,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "SafetyCopilot error: "
                f"{str(exc)}"
            ),
        ) from exc


# =========================================================
# SHUTDOWN
# =========================================================

@app.on_event("shutdown")
def shutdown_event():
    """
    Release persistent retrieval resources when FastAPI
    shuts down.
    """

    try:
        knowledge_base.close()
    except Exception:
        pass