import {
  Activity,
  Bot,
  Check,
  FileText,
  MessageSquare,
  Pencil,
  Plus,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";

import {
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  KeyboardEvent,
  MouseEvent,
} from "react";

import type {
  SavedConversation,
} from "../App";

// =====================================================
// VIEW TYPES
// =====================================================

export type AppView =
  | "assistant"
  | "knowledge-base"
  | "grounded-responses";

// =====================================================
// PROPS
// =====================================================

interface SidebarProps {
  sessionId: string;

  conversations: SavedConversation[];

  onNewChat: () => void;

  onSelectConversation: (
    conversationId: string
  ) => void;

  onRenameConversation: (
    conversationId: string,
    newTitle: string
  ) => void;

  onDeleteConversation: (
    conversationId: string
  ) => void;

  onClearAllConversations:
    () => void;

  activeView: AppView;

  onViewChange: (
    view: AppView
  ) => void;
}

// =====================================================
// DATE FORMATTER
// =====================================================

function formatConversationTime(
  value: string
) {
  try {
    const date = new Date(value);
    const now = new Date();

    const isToday =
      date.toDateString() ===
      now.toDateString();

    if (isToday) {
      return date.toLocaleTimeString(
        [],
        {
          hour: "2-digit",
          minute: "2-digit",
        }
      );
    }

    return date.toLocaleDateString(
      [],
      {
        month: "short",
        day: "numeric",
      }
    );
  } catch {
    return "";
  }
}

// =====================================================
// SIDEBAR
// =====================================================

function Sidebar({
  sessionId,
  conversations,
  onNewChat,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onClearAllConversations,
  activeView,
  onViewChange,
}: SidebarProps) {
  // ---------------------------------------------------
  // Rename state
  // ---------------------------------------------------

  const [
    editingConversationId,
    setEditingConversationId,
  ] = useState<string | null>(
    null
  );

  const [
    renameValue,
    setRenameValue,
  ] = useState("");

  const renameInputRef =
    useRef<HTMLInputElement | null>(
      null
    );

  // ---------------------------------------------------
  // Clear all confirmation
  // ---------------------------------------------------

  const [
    showClearConfirmation,
    setShowClearConfirmation,
  ] = useState(false);

  // ---------------------------------------------------
  // Focus rename input
  // ---------------------------------------------------

  useEffect(() => {
    if (!editingConversationId) {
      return;
    }

    renameInputRef.current?.focus();
    renameInputRef.current?.select();
  }, [editingConversationId]);

  // ===================================================
  // NEW CONVERSATION
  // ===================================================

  const handleNewConversation =
    () => {
      setEditingConversationId(
        null
      );

      setShowClearConfirmation(
        false
      );

      onNewChat();

      onViewChange(
        "assistant"
      );
    };

  // ===================================================
  // START RENAME
  // ===================================================

  const handleStartRename = (
    event:
      MouseEvent<HTMLButtonElement>,
    conversation:
      SavedConversation
  ) => {
    event.stopPropagation();

    setShowClearConfirmation(
      false
    );

    setEditingConversationId(
      conversation.id
    );

    setRenameValue(
      conversation.title
    );
  };

  // ===================================================
  // CANCEL RENAME
  // ===================================================

  const handleCancelRename =
    () => {
      setEditingConversationId(
        null
      );

      setRenameValue("");
    };

  // ===================================================
  // SAVE RENAME
  // ===================================================

  const handleSaveRename = (
    conversationId: string
  ) => {
    const cleanTitle =
      renameValue
        .replace(/\s+/g, " ")
        .trim();

    if (!cleanTitle) {
      return;
    }

    onRenameConversation(
      conversationId,
      cleanTitle
    );

    setEditingConversationId(
      null
    );

    setRenameValue("");
  };

  // ===================================================
  // RENAME KEYBOARD
  // ===================================================

  const handleRenameKeyDown = (
    event:
      KeyboardEvent<HTMLInputElement>,
    conversationId: string
  ) => {
    if (
      event.key === "Enter"
    ) {
      event.preventDefault();

      handleSaveRename(
        conversationId
      );

      return;
    }

    if (
      event.key === "Escape"
    ) {
      event.preventDefault();

      handleCancelRename();
    }
  };

  // ===================================================
  // DELETE
  // ===================================================

  const handleDelete = (
    event:
      MouseEvent<HTMLButtonElement>,
    conversationId: string
  ) => {
    event.stopPropagation();

    if (
      editingConversationId ===
      conversationId
    ) {
      handleCancelRename();
    }

    onDeleteConversation(
      conversationId
    );
  };

  // ===================================================
  // CLEAR ALL
  // ===================================================

  const handleShowClearAll =
    () => {
      setEditingConversationId(
        null
      );

      setShowClearConfirmation(
        true
      );
    };

  const handleCancelClearAll =
    () => {
      setShowClearConfirmation(
        false
      );
    };

  const handleConfirmClearAll =
    () => {
      setShowClearConfirmation(
        false
      );

      setEditingConversationId(
        null
      );

      onClearAllConversations();
    };

  // ===================================================
  // RENDER
  // ===================================================

  return (
    <aside className="sidebar">
      {/* =============================================
          BRAND
      ============================================== */}

      <div className="sidebar-brand">
        <div className="brand-icon">
          <ShieldCheck
            size={25}
          />
        </div>

        <div>
          <h1>
            SafetyCopilot
          </h1>

          <span>
            Industrial HSE Assistant
          </span>
        </div>
      </div>

      {/* =============================================
          NEW CONVERSATION
      ============================================== */}

      <button
        className="new-chat-button"
        onClick={
          handleNewConversation
        }
        type="button"
      >
        <Plus size={18} />

        <span>
          New Conversation
        </span>
      </button>

      {/* =============================================
          CONVERSATION HISTORY
      ============================================== */}

      <div className="sidebar-section conversation-section">
        <div className="sidebar-section-heading">
          <div className="conversation-heading-left">
            <p className="sidebar-label">
              CONVERSATIONS
            </p>

            {conversations.length >
              0 && (
              <span className="conversation-count">
                {
                  conversations.length
                }
              </span>
            )}
          </div>

          {conversations.length >
            0 &&
            !showClearConfirmation && (
              <button
                type="button"
                className="clear-conversations-button"
                onClick={
                  handleShowClearAll
                }
              >
                Clear All
              </button>
            )}
        </div>

        {/* -----------------------------------------
            CLEAR ALL CONFIRMATION
        ------------------------------------------ */}

        {showClearConfirmation && (
          <div className="clear-conversations-confirmation">
            <div className="clear-confirmation-copy">
              <strong>
                Clear all conversations?
              </strong>

              <span>
                This removes saved chat
                history from this browser.
              </span>
            </div>

            <div className="clear-confirmation-actions">
              <button
                type="button"
                className="clear-cancel-button"
                onClick={
                  handleCancelClearAll
                }
              >
                Cancel
              </button>

              <button
                type="button"
                className="clear-confirm-button"
                onClick={
                  handleConfirmClearAll
                }
              >
                <Trash2
                  size={13}
                />

                Clear
              </button>
            </div>
          </div>
        )}

        {/* -----------------------------------------
            EMPTY SESSION
        ------------------------------------------ */}

        {conversations.length ===
          0 && (
          <button
            type="button"
            className="session-card active"
            onClick={() =>
              onViewChange(
                "assistant"
              )
            }
          >
            <MessageSquare
              size={17}
            />

            <div className="session-info">
              <strong>
                New HSE Consultation
              </strong>

              <span>
                Current session
              </span>
            </div>
          </button>
        )}

        {/* -----------------------------------------
            SAVED CONVERSATIONS
        ------------------------------------------ */}

        {conversations.length >
          0 && (
          <div className="conversation-list">
            {conversations.map(
              (conversation) => {
                const isActive =
                  conversation.id ===
                    sessionId &&
                  activeView ===
                    "assistant";

                const isEditing =
                  editingConversationId ===
                  conversation.id;

                return (
                  <div
                    key={
                      conversation.id
                    }
                    className={`conversation-item ${
                      isActive
                        ? "active"
                        : ""
                    } ${
                      isEditing
                        ? "editing"
                        : ""
                    }`}
                    role="button"
                    tabIndex={
                      isEditing
                        ? -1
                        : 0
                    }
                    onClick={() => {
                      if (
                        isEditing
                      ) {
                        return;
                      }

                      onSelectConversation(
                        conversation.id
                      );
                    }}
                    onKeyDown={(
                      event
                    ) => {
                      if (
                        isEditing
                      ) {
                        return;
                      }

                      if (
                        event.key ===
                          "Enter" ||
                        event.key ===
                          " "
                      ) {
                        event.preventDefault();

                        onSelectConversation(
                          conversation.id
                        );
                      }
                    }}
                  >
                    <div className="conversation-main">
                      <div className="conversation-icon">
                        <MessageSquare
                          size={16}
                        />
                      </div>

                      {isEditing ? (
                        <div
                          className="conversation-rename-area"
                          onClick={(
                            event
                          ) =>
                            event.stopPropagation()
                          }
                        >
                          <input
                            ref={
                              renameInputRef
                            }
                            className="conversation-rename-input"
                            value={
                              renameValue
                            }
                            maxLength={
                              60
                            }
                            aria-label="Rename conversation"
                            onChange={(
                              event
                            ) =>
                              setRenameValue(
                                event.target
                                  .value
                              )
                            }
                            onKeyDown={(
                              event
                            ) =>
                              handleRenameKeyDown(
                                event,
                                conversation.id
                              )
                            }
                          />

                          <div className="conversation-rename-actions">
                            <button
                              type="button"
                              className="rename-action save"
                              title="Save name"
                              aria-label="Save conversation name"
                              disabled={
                                !renameValue.trim()
                              }
                              onClick={(
                                event
                              ) => {
                                event.stopPropagation();

                                handleSaveRename(
                                  conversation.id
                                );
                              }}
                            >
                              <Check
                                size={13}
                              />
                            </button>

                            <button
                              type="button"
                              className="rename-action cancel"
                              title="Cancel"
                              aria-label="Cancel rename"
                              onClick={(
                                event
                              ) => {
                                event.stopPropagation();

                                handleCancelRename();
                              }}
                            >
                              <X
                                size={13}
                              />
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="conversation-details">
                          <strong
                            title={
                              conversation.title
                            }
                          >
                            {
                              conversation.title
                            }
                          </strong>

                          <span>
                            {formatConversationTime(
                              conversation.updatedAt
                            )}
                          </span>
                        </div>
                      )}
                    </div>

                    {!isEditing && (
                      <div className="conversation-item-actions">
                        <button
                          type="button"
                          className="conversation-edit"
                          aria-label={`Rename ${conversation.title}`}
                          title="Rename conversation"
                          onClick={(
                            event
                          ) =>
                            handleStartRename(
                              event,
                              conversation
                            )
                          }
                        >
                          <Pencil
                            size={13}
                          />
                        </button>

                        <button
                          type="button"
                          className="conversation-delete"
                          aria-label={`Delete ${conversation.title}`}
                          title="Delete conversation"
                          onClick={(
                            event
                          ) =>
                            handleDelete(
                              event,
                              conversation.id
                            )
                          }
                        >
                          <Trash2
                            size={14}
                          />
                        </button>
                      </div>
                    )}
                  </div>
                );
              }
            )}
          </div>
        )}

        {/* -----------------------------------------
            NEW UNSAVED ACTIVE SESSION
        ------------------------------------------ */}

        {conversations.length >
          0 &&
          !conversations.some(
            (conversation) =>
              conversation.id ===
              sessionId
          ) && (
            <button
              type="button"
              className={`session-card ${
                activeView ===
                "assistant"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                onViewChange(
                  "assistant"
                )
              }
            >
              <MessageSquare
                size={17}
              />

              <div className="session-info">
                <strong>
                  New HSE Consultation
                </strong>

                <span>
                  Current session
                </span>
              </div>
            </button>
          )}
      </div>

      {/* =============================================
          KNOWLEDGE SYSTEM
      ============================================== */}

      <div className="sidebar-section knowledge-system-section">
        <p className="sidebar-label">
          KNOWLEDGE SYSTEM
        </p>

        <button
          type="button"
          className={`sidebar-item ${
            activeView ===
            "knowledge-base"
              ? "active"
              : ""
          }`}
          onClick={() =>
            onViewChange(
              "knowledge-base"
            )
          }
        >
          <FileText
            size={17}
          />

          <span>
            HSE Knowledge Base
          </span>
        </button>

        <button
          type="button"
          className={`sidebar-item ${
            activeView ===
            "assistant"
              ? "active"
              : ""
          }`}
          onClick={() =>
            onViewChange(
              "assistant"
            )
          }
        >
          <Bot size={17} />

          <span>
            RAG Assistant
          </span>
        </button>

        <button
          type="button"
          className={`sidebar-item ${
            activeView ===
            "grounded-responses"
              ? "active"
              : ""
          }`}
          onClick={() =>
            onViewChange(
              "grounded-responses"
            )
          }
        >
          <Activity
            size={17}
          />

          <span>
            Grounded Responses
          </span>
        </button>
      </div>

      {/* =============================================
          FOOTER
      ============================================== */}

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-dot" />

          <div>
            <strong>
              SafetyCopilot
            </strong>

            <span>
              Knowledge system ready
            </span>
          </div>
        </div>

        <p>
          AI-assisted HSE knowledge
          retrieval
        </p>
      </div>
    </aside>
  );
}

export default Sidebar;