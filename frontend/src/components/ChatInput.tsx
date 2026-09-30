import {
  ArrowUp,
  LoaderCircle,
  ShieldCheck,
} from "lucide-react";

import {
  type FormEvent,
  type KeyboardEvent,
  useState,
} from "react";

interface ChatInputProps {
  onSend: (question: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

function ChatInput({
  onSend,
  isLoading,
  disabled = false,
}: ChatInputProps) {
  const [question, setQuestion] =
    useState("");

  const submitQuestion = () => {
    const trimmedQuestion =
      question.trim();

    if (
      !trimmedQuestion ||
      isLoading ||
      disabled
    ) {
      return;
    }

    onSend(trimmedQuestion);
    setQuestion("");
  };

  const handleSubmit = (
    event: FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    submitQuestion();
  };

  const handleKeyDown = (
    event: KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      submitQuestion();
    }
  };

  return (
    <div className="chat-input-area">
      <form
        className="chat-input-container"
        onSubmit={handleSubmit}
      >
        <textarea
          value={question}
          onChange={(event) =>
            setQuestion(
              event.target.value
            )
          }
          onKeyDown={handleKeyDown}
          placeholder={
            disabled
              ? "SafetyCopilot API is offline..."
              : "Ask a question about workplace health and safety..."
          }
          rows={1}
          disabled={
            disabled || isLoading
          }
          aria-label="Ask SafetyCopilot"
        />

        <button
          className="send-button"
          type="submit"
          disabled={
            disabled ||
            isLoading ||
            !question.trim()
          }
          aria-label="Send question"
        >
          {isLoading ? (
            <LoaderCircle
              className="spinner"
              size={20}
            />
          ) : (
            <ArrowUp size={20} />
          )}
        </button>
      </form>

      <div className="input-footer">
        <div>
          <ShieldCheck size={13} />
          <span>
            Answers are grounded in the
            connected HSE knowledge base.
          </span>
        </div>

        <span className="input-hint">
          Enter to send · Shift + Enter
          for new line
        </span>
      </div>
    </div>
  );
}

export default ChatInput;