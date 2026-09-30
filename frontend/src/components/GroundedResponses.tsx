import {
  Activity,
  CheckCircle2,
  ExternalLink,
  FileText,
  ShieldCheck,
} from "lucide-react";

import ReactMarkdown from "react-markdown";

import type {
  ChatMessage as ChatMessageType,
} from "../types/api";

import { API_BASE_URL } from "../services/api";

interface GroundedResponsesProps {
  messages: ChatMessageType[];
}

function formatCategory(category: string) {
  return category
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}

function GroundedResponses({
  messages,
}: GroundedResponsesProps) {
  const groundedMessages = messages.filter(
    (message) =>
      message.role === "assistant" &&
      message.grounded === true
  );

  const openSource = (
    filename: string,
    page: number
  ) => {
    const encodedFilename =
      encodeURIComponent(filename);

    window.open(
      `${API_BASE_URL}/sources/${encodedFilename}#page=${page}`,
      "_blank",
      "noopener,noreferrer"
    );
  };

  return (
    <div className="grounded-page">
      {/* =====================================
          PAGE HEADER
      ====================================== */}

      <div className="grounded-page-header">
        <div className="grounded-page-icon">
          <Activity size={25} />
        </div>

        <div className="grounded-header-copy">
          <span className="grounded-eyebrow">
            VERIFIED RAG OUTPUT
          </span>

          <h1>Grounded Responses</h1>

          <p>
            Review answers generated from
            verified HSE documents with
            traceable source references.
          </p>
        </div>
      </div>

      {/* =====================================
          SUMMARY
      ====================================== */}

      <div className="grounded-summary">
        <div className="grounded-summary-icon">
          <ShieldCheck size={20} />
        </div>

        <div className="grounded-summary-copy">
          <strong>
            {groundedMessages.length}
          </strong>

          <span>
            grounded{" "}
            {groundedMessages.length === 1
              ? "response"
              : "responses"}{" "}
            in this session
          </span>
        </div>
      </div>

      {/* =====================================
          EMPTY STATE
      ====================================== */}

      {groundedMessages.length === 0 && (
        <div className="grounded-empty">
          <div className="grounded-empty-icon">
            <ShieldCheck size={30} />
          </div>

          <h2>
            No grounded responses yet
          </h2>

          <p>
            Ask SafetyCopilot an HSE question.
            Answers supported by the connected
            HSE knowledge base will appear here.
          </p>
        </div>
      )}

      {/* =====================================
          RESPONSE LIST
      ====================================== */}

      {groundedMessages.length > 0 && (
        <div className="grounded-response-list">
          {groundedMessages.map(
            (message, index) => (
              <article
                className="grounded-response-card"
                key={message.id}
              >
                {/* =========================
                    CARD HEADER
                ========================== */}

                <div className="grounded-card-header">
                  <div className="grounded-card-title">
                    <div className="grounded-number">
                      {String(
                        index + 1
                      ).padStart(2, "0")}
                    </div>

                    <div className="grounded-title-copy">
                      <span>
                        GROUNDED RESPONSE
                      </span>

                      <strong>
                        SafetyCopilot
                      </strong>
                    </div>
                  </div>

                  <div className="verified-badge">
                    <CheckCircle2 size={15} />
                    <span>Verified</span>
                  </div>
                </div>

                {/* =========================
                    ANSWER
                ========================== */}

                <div className="grounded-answer markdown-content">
                  <ReactMarkdown>
                    {message.content}
                  </ReactMarkdown>
                </div>

                {/* =========================
                    SOURCES
                ========================== */}

                {message.sources &&
                  message.sources.length >
                    0 && (
                    <div className="grounded-sources">
                      <div className="grounded-sources-heading">
                        <div className="grounded-sources-heading-icon">
                          <FileText
                            size={17}
                          />
                        </div>

                        <div>
                          <strong>
                            Verified HSE Sources
                          </strong>

                          <span>
                            Click a source to open
                            the referenced PDF page.
                          </span>
                        </div>
                      </div>

                      <div className="grounded-source-grid">
                        {message.sources.map(
                          (
                            source,
                            sourceIndex
                          ) => (
                            <button
                              type="button"
                              className="grounded-source-card"
                              key={`${source.filename}-${source.page}-${sourceIndex}`}
                              onClick={() =>
                                openSource(
                                  source.filename,
                                  source.page
                                )
                              }
                            >
                              <div className="grounded-source-number">
                                {sourceIndex +
                                  1}
                              </div>

                              <div className="grounded-source-info">
                                <strong>
                                  {
                                    source.filename
                                  }
                                </strong>

                                <div className="grounded-source-meta">
                                  <span>
                                    Page{" "}
                                    {
                                      source.page
                                    }
                                  </span>

                                  {source.category && (
                                    <span className="grounded-source-category">
                                      {formatCategory(
                                        source.category
                                      )}
                                    </span>
                                  )}
                                </div>
                              </div>

                              <ExternalLink
                                className="grounded-source-open"
                                size={17}
                              />
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  )}

                {/* =========================
                    MODEL INFO
                ========================== */}

                {(message.provider ||
                  message.model) && (
                  <div className="grounded-model-info">
                    {message.provider && (
                      <span>
                        Provider
                        <strong>
                          {
                            message.provider
                          }
                        </strong>
                      </span>
                    )}

                    {message.model && (
                      <span>
                        Model
                        <strong>
                          {message.model}
                        </strong>
                      </span>
                    )}
                  </div>
                )}
              </article>
            )
          )}
        </div>
      )}
    </div>
  );
}

export default GroundedResponses;