import {
  ExternalLink,
  FileText,
  Library,
  LoaderCircle,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  FormEvent,
} from "react";

import {
  deleteDocument,
  getDocuments,
  getDocumentUrl,
  uploadDocument,
} from "../services/api";

import type {
  DocumentItem,
} from "../services/api";


function KnowledgeBase() {
  const [documents, setDocuments] =
    useState<DocumentItem[]>([]);

  const [isLoading, setIsLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const [success, setSuccess] =
    useState<string | null>(null);

  const [showUpload, setShowUpload] =
    useState(false);

  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [category, setCategory] =
    useState("");

  const [isUploading, setIsUploading] =
    useState(false);

  const [
    deletingFilename,
    setDeletingFilename,
  ] = useState<string | null>(null);

  const fileInputRef =
    useRef<HTMLInputElement | null>(
      null
    );

  // -------------------------------------------------------
  // Load Documents
  // -------------------------------------------------------

  const loadDocuments =
    useCallback(async () => {
      try {
        setIsLoading(true);
        setError(null);

        const data =
          await getDocuments();

        setDocuments(data);
      } catch (err) {
        console.error(
          "Document loading failed:",
          err
        );

        setError(
          err instanceof Error
            ? err.message
            : "Unable to load documents."
        );
      } finally {
        setIsLoading(false);
      }
    }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  // -------------------------------------------------------
  // Open Document
  // -------------------------------------------------------

  const openDocument = (
    filename: string
  ) => {
    window.open(
      getDocumentUrl(filename),
      "_blank",
      "noopener,noreferrer"
    );
  };

  // -------------------------------------------------------
  // File Selection
  // -------------------------------------------------------

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file =
      event.target.files?.[0];

    setError(null);
    setSuccess(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (
      !file.name
        .toLowerCase()
        .endsWith(".pdf")
    ) {
      setSelectedFile(null);

      setError(
        "Only PDF documents can be uploaded."
      );

      event.target.value = "";

      return;
    }

    setSelectedFile(file);
  };

  // -------------------------------------------------------
  // Reset Upload Form
  // -------------------------------------------------------

  const resetUploadForm = () => {
    setSelectedFile(null);
    setCategory("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // -------------------------------------------------------
  // Upload Document
  // -------------------------------------------------------

  const handleUpload = async (
    event: FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    setError(null);
    setSuccess(null);

    if (!selectedFile) {
      setError(
        "Please select a PDF document."
      );
      return;
    }

    if (!category.trim()) {
      setError(
        "Please enter a document category."
      );
      return;
    }

    try {
      setIsUploading(true);

      const result =
        await uploadDocument(
          selectedFile,
          category
        );

      setSuccess(
        `${result.filename} uploaded and indexed successfully.`
      );

      resetUploadForm();
      setShowUpload(false);

      await loadDocuments();
    } catch (err) {
      console.error(
        "Document upload failed:",
        err
      );

      setError(
        err instanceof Error
          ? err.message
          : "Document upload failed."
      );
    } finally {
      setIsUploading(false);
    }
  };

  // -------------------------------------------------------
  // Delete Document
  // -------------------------------------------------------

  const handleDelete = async (
    document: DocumentItem
  ) => {
    const confirmed =
      window.confirm(
        `Delete "${document.filename}" from the SafetyCopilot knowledge base?\n\nThis will also remove its indexed vectors.`
      );

    if (!confirmed) {
      return;
    }

    setError(null);
    setSuccess(null);

    try {
      setDeletingFilename(
        document.filename
      );

      const result =
        await deleteDocument(
          document.filename
        );

      setDocuments(
        (currentDocuments) =>
          currentDocuments.filter(
            (item) =>
              item.filename !==
              document.filename
          )
      );

      setSuccess(
        `${result.filename} deleted successfully. Knowledge base now contains ${result.documents} documents and ${result.vectors} vectors.`
      );
    } catch (err) {
      console.error(
        "Document deletion failed:",
        err
      );

      setError(
        err instanceof Error
          ? err.message
          : "Document deletion failed."
      );
    } finally {
      setDeletingFilename(null);
    }
  };

  // -------------------------------------------------------
  // UI
  // -------------------------------------------------------

  return (
    <div className="knowledge-base-page">

      {/* Header */}

      <div className="knowledge-base-header">
        <div className="knowledge-base-icon">
          <Library size={24} />
        </div>

        <div>
          <span className="knowledge-eyebrow">
            HSE KNOWLEDGE SYSTEM
          </span>

          <h1>
            HSE Knowledge Base
          </h1>

          <p>
            Browse, upload, and manage
            the verified safety documents
            connected to SafetyCopilot.
          </p>
        </div>
      </div>

      {/* Toolbar */}

      <div className="knowledge-toolbar">

        <div className="knowledge-summary">
          <strong>
            {documents.length}
          </strong>

          <span>
            verified HSE documents
            available
          </span>
        </div>

        <div className="knowledge-actions">

          <button
            type="button"
            className="knowledge-refresh-button"
            onClick={() =>
              void loadDocuments()
            }
            disabled={
              isLoading ||
              isUploading ||
              deletingFilename !== null
            }
          >
            <RefreshCw
              size={17}
              className={
                isLoading
                  ? "spin"
                  : ""
              }
            />

            Refresh
          </button>

          <button
            type="button"
            className="knowledge-upload-button"
            onClick={() => {
              setShowUpload(
                (current) =>
                  !current
              );

              setError(null);
              setSuccess(null);
            }}
            disabled={
              isUploading ||
              deletingFilename !== null
            }
          >
            {showUpload ? (
              <X size={18} />
            ) : (
              <Plus size={18} />
            )}

            {showUpload
              ? "Close"
              : "Upload Document"}
          </button>

        </div>
      </div>

      {/* Upload Panel */}

      {showUpload && (
        <form
          className="knowledge-upload-panel"
          onSubmit={handleUpload}
        >
          <div className="upload-panel-heading">
            <div className="upload-panel-icon">
              <Upload size={20} />
            </div>

            <div>
              <strong>
                Add HSE Document
              </strong>

              <span>
                Upload a PDF and
                SafetyCopilot will index
                it into the knowledge base.
              </span>
            </div>
          </div>

          <div className="upload-form-grid">

            <label className="upload-field">
              <span>
                PDF Document
              </span>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={
                  handleFileChange
                }
                disabled={isUploading}
              />

              {selectedFile && (
                <small>
                  Selected:{" "}
                  {selectedFile.name}
                </small>
              )}
            </label>

            <label className="upload-field">
              <span>
                Category
              </span>

              <input
                type="text"
                value={category}
                onChange={(event) =>
                  setCategory(
                    event.target.value
                  )
                }
                placeholder="e.g. Fire Safety"
                disabled={isUploading}
              />
            </label>

          </div>

          <div className="upload-panel-footer">
            <span className="upload-help">
              PDF files only. The document
              becomes searchable after
              indexing completes.
            </span>

            <button
              type="submit"
              className="upload-submit-button"
              disabled={
                isUploading ||
                !selectedFile ||
                !category.trim()
              }
            >
              {isUploading ? (
                <>
                  <LoaderCircle
                    size={18}
                    className="spin"
                  />

                  Uploading &
                  Indexing...
                </>
              ) : (
                <>
                  <Upload size={18} />

                  Upload & Index
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* Success */}

      {success && (
        <div className="knowledge-success">
          {success}
        </div>
      )}

      {/* Error */}

      {error && (
        <div className="knowledge-error">
          <strong>
            Knowledge Base Error
          </strong>

          <span>
            {error}
          </span>
        </div>
      )}

      {/* Loading */}

      {isLoading && (
        <div className="knowledge-state">
          <LoaderCircle
            size={20}
            className="spin"
          />

          Loading HSE documents...
        </div>
      )}

      {/* Empty */}

      {!isLoading &&
        !error &&
        documents.length === 0 && (
          <div className="knowledge-state">
            <FileText size={24} />

            <strong>
              No HSE documents
            </strong>

            <span>
              Upload a PDF to start
              building the SafetyCopilot
              knowledge base.
            </span>
          </div>
        )}

      {/* Documents */}

      {!isLoading &&
        documents.length > 0 && (
          <div className="document-grid">

            {documents.map(
              (document, index) => {
                const isDeleting =
                  deletingFilename ===
                  document.filename;

                return (
                  <div
                    className="document-card"
                    key={
                      document.filename
                    }
                  >
                    <button
                      type="button"
                      className="document-main-button"
                      onClick={() =>
                        openDocument(
                          document.filename
                        )
                      }
                      disabled={
                        isDeleting
                      }
                    >
                      <div className="document-icon">
                        <FileText
                          size={22}
                        />
                      </div>

                      <div className="document-content">
                        <span className="document-number">
                          DOCUMENT{" "}
                          {String(
                            index + 1
                          ).padStart(
                            2,
                            "0"
                          )}
                        </span>

                        <strong>
                          {
                            document.filename
                          }
                        </strong>

                        <span className="document-category">
                          {
                            document.category
                          }
                        </span>
                      </div>

                      <ExternalLink
                        className="document-open-icon"
                        size={18}
                      />
                    </button>

                    <button
                      type="button"
                      className="document-delete-button"
                      title={`Delete ${document.filename}`}
                      aria-label={`Delete ${document.filename}`}
                      disabled={
                        isDeleting ||
                        isUploading
                      }
                      onClick={() =>
                        void handleDelete(
                          document
                        )
                      }
                    >
                      {isDeleting ? (
                        <LoaderCircle
                          size={17}
                          className="spin"
                        />
                      ) : (
                        <Trash2
                          size={17}
                        />
                      )}

                      <span>
                        {isDeleting
                          ? "Deleting..."
                          : "Delete"}
                      </span>
                    </button>
                  </div>
                );
              }
            )}

          </div>
        )}

    </div>
  );
}

export default KnowledgeBase;