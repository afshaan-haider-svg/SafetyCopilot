import {
  useEffect,
  useRef,
  useState,
} from "react";

import Sidebar, {
  type AppView,
} from "./components/Sidebar";

import Header from "./components/Header";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import WelcomePanel from "./components/WelcomePanel";
import KnowledgeBase from "./components/KnowledgeBase";
import GroundedResponses from "./components/GroundedResponses";

import {
  askSafetyCopilot,
  checkHealth,
} from "./services/api";

import type {
  ChatMessage as ChatMessageType,
  HealthResponse,
} from "./types/api";

import "./App.css";

// =====================================================
// TYPES
// =====================================================

export interface SavedConversation {
  id: string;
  title: string;
  messages: ChatMessageType[];
  createdAt: string;
  updatedAt: string;
  isCustomTitle?: boolean;
}

// =====================================================
// STORAGE
// =====================================================

const STORAGE_KEY =
  "safetycopilot-conversations";

const ACTIVE_SESSION_KEY =
  "safetycopilot-active-session";

// =====================================================
// HELPERS
// =====================================================

function createSessionId() {
  return `safety-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

function createConversationTitle(
  messages: ChatMessageType[]
) {
  const firstUserMessage =
    messages.find(
      (message) =>
        message.role === "user"
    );

  if (!firstUserMessage) {
    return "New HSE Consultation";
  }

  const cleanTitle =
    firstUserMessage.content
      .replace(/\s+/g, " ")
      .trim();

  if (!cleanTitle) {
    return "HSE Consultation";
  }

  if (cleanTitle.length <= 38) {
    return cleanTitle;
  }

  return `${cleanTitle.slice(0, 38)}...`;
}

function restoreMessages(
  messages: ChatMessageType[]
): ChatMessageType[] {
  return messages.map(
    (message) => ({
      ...message,
      timestamp: new Date(
        message.timestamp
      ),
    })
  );
}

function loadConversations():
  SavedConversation[] {
  try {
    const saved =
      localStorage.getItem(
        STORAGE_KEY
      );

    if (!saved) {
      return [];
    }

    const parsed =
      JSON.parse(saved);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .filter(
        (conversation) =>
          conversation &&
          typeof conversation.id ===
            "string" &&
          Array.isArray(
            conversation.messages
          )
      )
      .map(
        (conversation) => ({
          id: conversation.id,

          title:
            typeof conversation.title ===
            "string"
              ? conversation.title
              : "HSE Consultation",

          messages: restoreMessages(
            conversation.messages
          ),

          createdAt:
            conversation.createdAt ??
            new Date().toISOString(),

          updatedAt:
            conversation.updatedAt ??
            new Date().toISOString(),

          isCustomTitle:
            conversation.isCustomTitle ===
            true,
        })
      );
  } catch (error) {
    console.error(
      "Failed to load SafetyCopilot conversations:",
      error
    );

    return [];
  }
}

function loadActiveSessionId() {
  try {
    return localStorage.getItem(
      ACTIVE_SESSION_KEY
    );
  } catch {
    return null;
  }
}

// =====================================================
// APP
// =====================================================

function App() {
  // ---------------------------------------------------
  // Initial stored data
  // ---------------------------------------------------

  const initialConversationsRef =
    useRef<SavedConversation[]>(
      loadConversations()
    );

  const initialActiveSessionRef =
    useRef<string | null>(
      loadActiveSessionId()
    );

  const initialSessionRef =
    useRef<string>(
      (() => {
        const conversations =
          initialConversationsRef.current;

        const storedActive =
          initialActiveSessionRef.current;

        if (
          storedActive &&
          conversations.some(
            (conversation) =>
              conversation.id ===
              storedActive
          )
        ) {
          return storedActive;
        }

        if (
          conversations.length > 0
        ) {
          return conversations[0].id;
        }

        return createSessionId();
      })()
    );

  // ---------------------------------------------------
  // Conversations
  // ---------------------------------------------------

  const [
    conversations,
    setConversations,
  ] = useState<SavedConversation[]>(
    initialConversationsRef.current
  );

  // ---------------------------------------------------
  // Active session
  // ---------------------------------------------------

  const [
    sessionId,
    setSessionId,
  ] = useState<string>(
    initialSessionRef.current
  );

  // ---------------------------------------------------
  // Messages
  // ---------------------------------------------------

  const [
    messages,
    setMessages,
  ] = useState<
    ChatMessageType[]
  >(() => {
    const conversation =
      initialConversationsRef.current.find(
        (item) =>
          item.id ===
          initialSessionRef.current
      );

    return (
      conversation?.messages ?? []
    );
  });

  // ---------------------------------------------------
  // Backend status
  // ---------------------------------------------------

  const [
    health,
    setHealth,
  ] = useState<
    HealthResponse | null
  >(null);

  const [
    isOnline,
    setIsOnline,
  ] = useState(false);

  // ---------------------------------------------------
  // UI state
  // ---------------------------------------------------

  const [
    isLoading,
    setIsLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(null);

  // NEW — stores the most recent failed question
  // so the user can retry it.
  const [
    lastFailedQuestion,
    setLastFailedQuestion,
  ] = useState<
    string | null
  >(null);

  const [
    activeView,
    setActiveView,
  ] = useState<AppView>(
    "assistant"
  );

  const messagesEndRef =
    useRef<HTMLDivElement | null>(
      null
    );

  // ===================================================
  // SAVE CONVERSATIONS
  // ===================================================

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(
          conversations
        )
      );
    } catch (error) {
      console.error(
        "Failed to save conversations:",
        error
      );
    }
  }, [conversations]);

  // ===================================================
  // SAVE ACTIVE SESSION
  // ===================================================

  useEffect(() => {
    try {
      localStorage.setItem(
        ACTIVE_SESSION_KEY,
        sessionId
      );
    } catch (error) {
      console.error(
        "Failed to save active session:",
        error
      );
    }
  }, [sessionId]);

  // ===================================================
  // SYNC ACTIVE CONVERSATION
  // ===================================================

  useEffect(() => {
    if (messages.length === 0) {
      return;
    }

    setConversations(
      (current) => {
        const existing =
          current.find(
            (conversation) =>
              conversation.id ===
              sessionId
          );

        const now =
          new Date().toISOString();

        if (!existing) {
          const newConversation:
            SavedConversation = {
              id: sessionId,

              title:
                createConversationTitle(
                  messages
                ),

              messages,

              createdAt: now,
              updatedAt: now,

              isCustomTitle: false,
            };

          return [
            newConversation,
            ...current,
          ];
        }

        const updatedConversation:
          SavedConversation = {
            ...existing,

            title:
              existing.isCustomTitle
                ? existing.title
                : createConversationTitle(
                    messages
                  ),

            messages,

            updatedAt: now,
          };

        return [
          updatedConversation,

          ...current.filter(
            (conversation) =>
              conversation.id !==
              sessionId
          ),
        ];
      }
    );
  }, [
    messages,
    sessionId,
  ]);

  // ===================================================
  // BACKEND HEALTH CHECK
  // ===================================================

  useEffect(() => {
    let mounted = true;

    const loadHealth =
      async () => {
        try {
          const data =
            await checkHealth();

          if (!mounted) {
            return;
          }

          setHealth(data);
          setIsOnline(true);
        } catch (err) {
          console.error(
            "SafetyCopilot health check failed:",
            err
          );

          if (!mounted) {
            return;
          }

          setIsOnline(false);
        }
      };

    loadHealth();

    const interval =
      window.setInterval(
        loadHealth,
        30000
      );

    return () => {
      mounted = false;

      window.clearInterval(
        interval
      );
    };
  }, []);

  // ===================================================
  // AUTO SCROLL
  // ===================================================

  useEffect(() => {
    if (
      activeView !== "assistant"
    ) {
      return;
    }

    messagesEndRef.current
      ?.scrollIntoView({
        behavior: "smooth",
      });
  }, [
    messages,
    isLoading,
    activeView,
  ]);

  // ===================================================
  // SEND QUESTION
  // ===================================================

  const handleSendQuestion =
    async (
      question: string,
      isRetry = false
    ) => {
      const cleanQuestion =
        question.trim();

      if (
        !cleanQuestion ||
        isLoading
      ) {
        return;
      }

      setActiveView(
        "assistant"
      );

      setError(null);

      // A fresh request clears the previous
      // retry state. If it fails again, the
      // catch block will restore it.
      setLastFailedQuestion(null);

      // On retry we do not add the same user
      // question to the conversation twice.
      if (!isRetry) {
        const userMessage:
          ChatMessageType = {
            id: crypto.randomUUID(),
            role: "user",
            content: cleanQuestion,
            timestamp: new Date(),
          };

        setMessages(
          (current) => [
            ...current,
            userMessage,
          ]
        );
      }

      setIsLoading(true);

      try {
        const response =
          await askSafetyCopilot({
            question:
              cleanQuestion,

            session_id:
              sessionId,
          });

        const assistantMessage:
          ChatMessageType = {
            id: crypto.randomUUID(),

            role: "assistant",

            content:
              response.answer,

            timestamp:
              new Date(),

            sources:
              response.sources,

            grounded:
              response.grounded,

            provider:
              response.provider,

            model:
              response.model,
          };

        setMessages(
          (current) => [
            ...current,
            assistantMessage,
          ]
        );

        // Request succeeded.
        setError(null);
        setLastFailedQuestion(null);

        // Successful request confirms
        // backend connectivity.
        setIsOnline(true);
      } catch (err) {
        console.error(
          "SafetyCopilot request failed:",
          err
        );

        const message =
          err instanceof Error
            ? err.message
            : (
                "SafetyCopilot request failed. " +
                "Please try again."
              );

        setError(message);

        // Remember failed question for Retry.
        setLastFailedQuestion(
          cleanQuestion
        );

        // Do not mark FastAPI offline for
        // retrieval, generation, validation,
        // Groq, or other processing errors.
        if (
          message ===
          "Unable to connect to SafetyCopilot API."
        ) {
          setIsOnline(false);
        }
      } finally {
        setIsLoading(false);
      }
    };

  // ===================================================
  // RETRY FAILED QUESTION
  // ===================================================

  const handleRetry = async () => {
    if (
      !lastFailedQuestion ||
      isLoading
    ) {
      return;
    }

    // If the previous error was a connection
    // failure, check the backend first.
    if (!isOnline) {
      try {
        const data =
          await checkHealth();

        setHealth(data);
        setIsOnline(true);
      } catch (err) {
        console.error(
          "SafetyCopilot retry health check failed:",
          err
        );

        setIsOnline(false);

        setError(
          "Unable to connect to SafetyCopilot API."
        );

        return;
      }
    }

    await handleSendQuestion(
      lastFailedQuestion,
      true
    );
  };

  // ===================================================
  // NEW CONVERSATION
  // ===================================================

  const handleNewChat = () => {
    const newSessionId =
      createSessionId();

    setSessionId(
      newSessionId
    );

    setMessages([]);

    setError(null);

    setLastFailedQuestion(
      null
    );

    setIsLoading(false);

    setActiveView(
      "assistant"
    );
  };

  // ===================================================
  // OPEN SAVED CONVERSATION
  // ===================================================

  const handleSelectConversation = (
    conversationId: string
  ) => {
    if (
      isLoading ||
      conversationId ===
        sessionId
    ) {
      return;
    }

    const conversation =
      conversations.find(
        (item) =>
          item.id ===
          conversationId
      );

    if (!conversation) {
      return;
    }

    setSessionId(
      conversation.id
    );

    setMessages(
      restoreMessages(
        conversation.messages
      )
    );

    setError(null);

    setLastFailedQuestion(
      null
    );

    setActiveView(
      "assistant"
    );
  };

  // ===================================================
  // RENAME CONVERSATION
  // ===================================================

  const handleRenameConversation = (
    conversationId: string,
    newTitle: string
  ) => {
    const cleanTitle =
      newTitle
        .replace(/\s+/g, " ")
        .trim();

    if (!cleanTitle) {
      return;
    }

    const finalTitle =
      cleanTitle.length > 60
        ? `${cleanTitle.slice(
            0,
            60
          )}...`
        : cleanTitle;

    setConversations(
      (current) =>
        current.map(
          (conversation) => {
            if (
              conversation.id !==
              conversationId
            ) {
              return conversation;
            }

            return {
              ...conversation,

              title:
                finalTitle,

              isCustomTitle:
                true,

              updatedAt:
                new Date().toISOString(),
            };
          }
        )
    );
  };

  // ===================================================
  // DELETE CONVERSATION
  // ===================================================

  const handleDeleteConversation = (
    conversationId: string
  ) => {
    const remaining =
      conversations.filter(
        (conversation) =>
          conversation.id !==
          conversationId
      );

    setConversations(
      remaining
    );

    if (
      conversationId !==
      sessionId
    ) {
      return;
    }

    if (remaining.length > 0) {
      const next =
        remaining[0];

      setSessionId(
        next.id
      );

      setMessages(
        restoreMessages(
          next.messages
        )
      );
    } else {
      setSessionId(
        createSessionId()
      );

      setMessages([]);
    }

    setError(null);

    setLastFailedQuestion(
      null
    );

    setIsLoading(false);

    setActiveView(
      "assistant"
    );
  };

  // ===================================================
  // CLEAR ALL CONVERSATIONS
  // ===================================================

  const handleClearAllConversations =
    () => {
      const newSessionId =
        createSessionId();

      setConversations([]);

      setSessionId(
        newSessionId
      );

      setMessages([]);

      setError(null);

      setLastFailedQuestion(
        null
      );

      setIsLoading(false);

      setActiveView(
        "assistant"
      );

      try {
        localStorage.removeItem(
          STORAGE_KEY
        );

        localStorage.setItem(
          ACTIVE_SESSION_KEY,
          newSessionId
        );
      } catch (error) {
        console.error(
          "Failed to clear SafetyCopilot conversations:",
          error
        );
      }
    };

  // ===================================================
  // NAVIGATION
  // ===================================================

  const handleViewChange = (
    view: AppView
  ) => {
    setActiveView(view);

    setError(null);

    setLastFailedQuestion(
      null
    );
  };

  // ===================================================
  // ERROR TYPE
  // ===================================================

  const isConnectionError =
    error ===
    "Unable to connect to SafetyCopilot API.";

  // ===================================================
  // MAIN UI
  // ===================================================

  return (
    <div className="app-shell">
      <Sidebar
        sessionId={
          sessionId
        }

        conversations={
          conversations
        }

        onNewChat={
          handleNewChat
        }

        onSelectConversation={
          handleSelectConversation
        }

        onRenameConversation={
          handleRenameConversation
        }

        onDeleteConversation={
          handleDeleteConversation
        }

        onClearAllConversations={
          handleClearAllConversations
        }

        activeView={
          activeView
        }

        onViewChange={
          handleViewChange
        }
      />

      <main className="main-workspace">
        <Header
          isOnline={
            isOnline
          }

          documents={
            health?.documents ?? 0
          }

          chunks={
            health?.chunks ?? 0
          }
        />

        {/* =================================
            HSE KNOWLEDGE BASE
        ================================= */}

        {activeView ===
          "knowledge-base" && (
          <section className="knowledge-workspace">
            <KnowledgeBase />
          </section>
        )}

        {/* =================================
            GROUNDED RESPONSES
        ================================= */}

        {activeView ===
          "grounded-responses" && (
          <section className="grounded-workspace">
            <GroundedResponses
              messages={
                messages
              }
            />
          </section>
        )}

        {/* =================================
            RAG ASSISTANT
        ================================= */}

        {activeView ===
          "assistant" && (
          <section className="chat-workspace">
            <div className="chat-scroll-area">
              {messages.length ===
              0 ? (
                <WelcomePanel
                  onQuestionSelect={
                    handleSendQuestion
                  }
                />
              ) : (
                <div className="messages-container">
                  {messages.map(
                    (message) => (
                      <ChatMessage
                        key={
                          message.id
                        }

                        message={
                          message
                        }
                      />
                    )
                  )}

                  {isLoading && (
                    <div className="thinking-message">
                      <div className="thinking-indicator">
                        <span />
                        <span />
                        <span />
                      </div>

                      <div>
                        <strong>
                          SafetyCopilot
                          is analyzing
                          HSE evidence
                        </strong>

                        <p>
                          Retrieving and
                          grounding the
                          response...
                        </p>
                      </div>
                    </div>
                  )}

                  <div
                    ref={
                      messagesEndRef
                    }
                  />
                </div>
              )}
            </div>

            {/* =================================
                PROFESSIONAL ERROR STATE
            ================================= */}

            {error && (
              <div
                className={`api-error ${
                  isConnectionError
                    ? "connection-error"
                    : "request-error"
                }`}
              >
                <div className="api-error-content">
                  <strong>
                    {isConnectionError
                      ? "Connection Error"
                      : "Request Error"}
                  </strong>

                  <span>
                    {error}
                  </span>
                </div>

                {lastFailedQuestion && (
                  <button
                    type="button"
                    className="retry-button"
                    onClick={
                      handleRetry
                    }
                    disabled={
                      isLoading
                    }
                  >
                    {isLoading
                      ? "Retrying..."
                      : "Retry"}
                  </button>
                )}
              </div>
            )}

            <ChatInput
              onSend={
                handleSendQuestion
              }

              isLoading={
                isLoading
              }

              disabled={
                !isOnline
              }
            />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;