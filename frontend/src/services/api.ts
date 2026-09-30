import type {
  AskRequest,
  AskResponse,
  HealthResponse,
} from "../types/api";

export const API_BASE_URL =
  "http://127.0.0.1:8000";

// ---------------------------------------------------------
// Document Types
// ---------------------------------------------------------

export interface DocumentItem {
  filename: string;
  category: string;
}

export interface DocumentUploadResponse {
  message: string;
  filename: string;
  category: string;
  documents: number;
  chunks: number;
  vectors: number;
}

export interface DocumentDeleteResponse {
  message: string;
  filename: string;
  documents: number;
  chunks: number;
  vectors: number;
}

// ---------------------------------------------------------
// Extract backend error message
// ---------------------------------------------------------

async function getErrorMessage(
  response: Response,
  fallbackMessage: string
): Promise<string> {
  try {
    const errorData =
      await response.json();

    if (
      errorData &&
      typeof errorData.detail ===
        "string"
    ) {
      return errorData.detail;
    }

    if (errorData?.detail) {
      return JSON.stringify(
        errorData.detail
      );
    }
  } catch {
    // Response may not contain JSON.
  }

  return fallbackMessage;
}

// ---------------------------------------------------------
// Health Check
// ---------------------------------------------------------

export async function checkHealth():
  Promise<HealthResponse> {
  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/health`
    );
  } catch {
    throw new Error(
      "Unable to connect to SafetyCopilot API."
    );
  }

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "SafetyCopilot health check failed."
      );

    throw new Error(message);
  }

  try {
    return await response.json();
  } catch {
    throw new Error(
      "SafetyCopilot returned an invalid health response."
    );
  }
}

// ---------------------------------------------------------
// Ask SafetyCopilot
// ---------------------------------------------------------

export async function askSafetyCopilot(
  request: AskRequest
): Promise<AskResponse> {
  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/ask`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",

          Accept:
            "application/json",
        },

        body: JSON.stringify(
          request
        ),
      }
    );
  } catch {
    throw new Error(
      "Unable to connect to SafetyCopilot API."
    );
  }

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "SafetyCopilot request failed. Please try again."
      );

    throw new Error(message);
  }

  try {
    return await response.json();
  } catch {
    throw new Error(
      "SafetyCopilot returned an invalid response."
    );
  }
}

// ---------------------------------------------------------
// Get Knowledge Base Documents
// ---------------------------------------------------------

export async function getDocuments():
  Promise<DocumentItem[]> {
  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/documents`,
      {
        method: "GET",

        headers: {
          Accept:
            "application/json",
        },
      }
    );
  } catch {
    throw new Error(
      "Unable to connect to SafetyCopilot API."
    );
  }

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "Unable to load HSE documents."
      );

    throw new Error(message);
  }

  try {
    return await response.json();
  } catch {
    throw new Error(
      "SafetyCopilot returned an invalid document list."
    );
  }
}

// ---------------------------------------------------------
// Upload Knowledge Base Document
// ---------------------------------------------------------

export async function uploadDocument(
  file: File,
  category: string
): Promise<DocumentUploadResponse> {
  if (!file) {
    throw new Error(
      "Please select a PDF document."
    );
  }

  if (
    !file.name
      .toLowerCase()
      .endsWith(".pdf")
  ) {
    throw new Error(
      "Only PDF documents can be uploaded."
    );
  }

  if (!category.trim()) {
    throw new Error(
      "Please enter a document category."
    );
  }

  const formData =
    new FormData();

  formData.append(
    "file",
    file
  );

  formData.append(
    "category",
    category.trim()
  );

  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/documents/upload`,
      {
        method: "POST",

        body: formData,
      }
    );
  } catch {
    throw new Error(
      "Unable to connect to SafetyCopilot API."
    );
  }

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "Document upload failed."
      );

    throw new Error(message);
  }

  try {
    return await response.json();
  } catch {
    throw new Error(
      "SafetyCopilot returned an invalid upload response."
    );
  }
}

// ---------------------------------------------------------
// Delete Knowledge Base Document
// ---------------------------------------------------------

export async function deleteDocument(
  filename: string
): Promise<DocumentDeleteResponse> {
  const safeFilename =
    filename.trim();

  if (!safeFilename) {
    throw new Error(
      "Document filename is required."
    );
  }

  const encodedFilename =
    encodeURIComponent(
      safeFilename
    );

  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/documents/${encodedFilename}`,
      {
        method: "DELETE",

        headers: {
          Accept:
            "application/json",
        },
      }
    );
  } catch {
    throw new Error(
      "Unable to connect to SafetyCopilot API."
    );
  }

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "Document deletion failed."
      );

    throw new Error(message);
  }

  try {
    return await response.json();
  } catch {
    throw new Error(
      "SafetyCopilot returned an invalid deletion response."
    );
  }
}

// ---------------------------------------------------------
// Source PDF URL
// ---------------------------------------------------------

export function getDocumentUrl(
  filename: string
): string {
  return (
    `${API_BASE_URL}/sources/` +
    encodeURIComponent(
      filename
    )
  );
}