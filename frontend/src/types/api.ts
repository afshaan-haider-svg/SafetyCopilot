export interface SourceItem {
  filename: string;
  page: number;
  category: string;
}

export interface AskRequest {
  question: string;
  session_id: string;
}

export interface AskResponse {
  question: string;
  session_id: string;
  answer: string;
  sources: SourceItem[];
  provider: string | null;
  model: string | null;
  grounded: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  documents: number;
  chunks: number;
}

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  sources?: SourceItem[];
  grounded?: boolean;
  provider?: string | null;
  model?: string | null;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}