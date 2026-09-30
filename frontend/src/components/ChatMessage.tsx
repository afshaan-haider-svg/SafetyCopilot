import {
  Bot,
  CheckCircle2,
  FileText,
  User,
} from "lucide-react";

import ReactMarkdown from "react-markdown";

import type {
  ChatMessage as ChatMessageType,
} from "../types/api";

interface ChatMessageProps {
  message: ChatMessageType;
}

function formatCategory(category: string) {
  return category
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}

function ChatMessage({
  message,
}: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={
        isUser
          ? "message-row user-message"
          : "message-row assistant-message"
      }
    >
      <div className="message-avatar">
        {isUser ? (
          <User size={19} />
        ) : (
          <Bot size={20} />
        )}
      </div>

      <div className="message-content">
        <div className="message-heading">
          <strong>
            {isUser ? "You" : "SafetyCopilot"}
          </strong>

          {!isUser && message.grounded && (
            <span className="grounded-badge">
              <CheckCircle2 size={14} />
              Grounded
            </span>
          )}
        </div>

        {isUser ? (
          <div className="message-text">
            {message.content}
          </div>
        ) : (
          <div className="message-text markdown-content">
            <ReactMarkdown>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser &&
          message.sources &&
          message.sources.length > 0 && (
            <div className="sources-section">
              <div className="sources-heading">
                <FileText size={16} />
                <span>Verified HSE Sources</span>
              </div>

              <div className="source-list">
                {message.sources.map(
                  (source, index) => (
                    <a
                      key={`${source.filename}-${source.page}-${index}`}
                      className="source-card source-card-link"
                      href={`http://127.0.0.1:8000/sources/${encodeURIComponent(
                        source.filename
                      )}#page=${source.page}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`Open ${source.filename} — Page ${source.page}`}
                    >
                      <div className="source-number">
                        {index + 1}
                      </div>

                      <div className="source-details">
                        <strong>
                          {source.filename}
                        </strong>

                        <div className="source-meta">
                          <span>
                            Page {source.page}
                          </span>

                          {source.category && (
                            <span className="source-category">
                              {formatCategory(
                                source.category
                              )}
                            </span>
                          )}
                        </div>
                      </div>
                    </a>
                  )
                )}
              </div>
            </div>
          )}

        {!isUser &&
          (message.provider || message.model) && (
            <div className="model-info">
              {message.provider && (
                <span>
                  Provider:{" "}
                  <strong>
                    {message.provider}
                  </strong>
                </span>
              )}

              {message.model && (
                <span>
                  Model:{" "}
                  <strong>
                    {message.model}
                  </strong>
                </span>
              )}
            </div>
          )}
      </div>
    </div>
  );
}

export default ChatMessage;