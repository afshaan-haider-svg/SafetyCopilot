"""
SafetyCopilot â€” Grounded RAG Generator

Supports:
- Google Gemini
- Groq
- Provider fallback
- Evidence-grounded answers
- Verified source metadata
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from groq import Groq


# ============================================================
# Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# RAG Generator
# ============================================================

class RAGGenerator:

    def __init__(
        self,
        gemini_api_key: str | None = None,
        groq_api_key: str | None = None,
        gemini_model: str | None = None,
        groq_model: str | None = None,
        gemini_retries: int = 2,
        retry_delay: int = 3,
    ) -> None:

        # ----------------------------------------------------
        # API keys
        #
        # None = read from .env
        # ""   = explicitly disable provider
        # ----------------------------------------------------

        if gemini_api_key is None:
            self.gemini_api_key = os.getenv(
                "GEMINI_API_KEY"
            )
        else:
            self.gemini_api_key = gemini_api_key

        if groq_api_key is None:
            self.groq_api_key = os.getenv(
                "GROQ_API_KEY"
            )
        else:
            self.groq_api_key = groq_api_key

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        self.gemini_model = (
            gemini_model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.5-flash-lite",
            )
        )

        self.groq_model = (
            groq_model
            or os.getenv(
                "GROQ_MODEL",
                "qwen/qwen3.8-27b",
            )
        )

        self.gemini_retries = gemini_retries
        self.retry_delay = retry_delay

        # ----------------------------------------------------
        # Clients
        # ----------------------------------------------------

        self.gemini_client = None
        self.groq_client = None

        if self.gemini_api_key:
            self.gemini_client = genai.Client(
                api_key=self.gemini_api_key
            )

        if self.groq_api_key:
            self.groq_client = Groq(
                api_key=self.groq_api_key
            )

        if (
            self.gemini_client is None
            and self.groq_client is None
        ):
            raise RuntimeError(
                "No LLM API key found. "
                "Add GEMINI_API_KEY or GROQ_API_KEY "
                "to the project .env file."
            )

    # ========================================================
    # Context
    # ========================================================

    def _build_context(
        self,
        evidence: List[Dict[str, Any]],
    ) -> str:

        blocks = []

        for number, item in enumerate(
            evidence,
            start=1,
        ):

            filename = item.get(
                "filename",
                "Unknown source",
            )

            page_number = item.get(
                "page_number",
                "?",
            )

            category = item.get(
                "category",
                "unknown",
            )

            text = item.get(
                "text",
                "",
            ).strip()

            block = (
                f"[EVIDENCE {number}]\n"
                f"Source: {filename}\n"
                f"Page: {page_number}\n"
                f"Category: {category}\n"
                f"Content:\n{text}"
            )

            blocks.append(block)

        return "\n\n".join(blocks)

    # ========================================================
    # Prompt
    # ========================================================

    def _build_prompt(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
    ) -> str:

        context = self._build_context(
            evidence
        )

        return f"""
You are SafetyCopilot, an industrial HSE knowledge assistant.

Answer the user's question ONLY from the supplied HSE evidence.

STRICT RULES:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent regulations, procedures, limits,
   equipment, legal requirements, or recommendations.
4. Support important safety claims with citations
   such as [1], [2], and [3].
5. Citation numbers must correspond exactly to the
   supplied EVIDENCE numbers.
6. If the supplied evidence is relevant to the user's question, you MUST answer
   using the useful information that is actually present in that evidence.
   A partial evidence-based answer is acceptable and preferred over refusing
   to answer. Clearly limit the answer to what the evidence supports.

7. For broad questions, summarize the relevant examples, requirements, and
   guidance found in the supplied evidence. Do not require the evidence to
   contain a single complete or exhaustive list before answering.

8. Use the insufficient-information response ONLY when the supplied evidence
   is genuinely unrelated to the user's question or contains no information
   that can help answer it. In that case reply exactly:
   "The available HSE documents do not provide enough information to answer this question reliably."

9. Do not invent filenames or page numbers.S
10. Do not create a separate sources section.
    The application will append verified sources.

USER QUESTION:
{question}

HSE EVIDENCE:
{context}

ANSWER:
""".strip()

    # ========================================================
    # Gemini
    # ========================================================

    def _call_gemini(
        self,
        prompt: str,
    ) -> str:

        if self.gemini_client is None:
            raise RuntimeError(
                "Gemini is not configured."
            )

        last_error = None

        for attempt in range(
            1,
            self.gemini_retries + 1,
        ):

            try:

                interaction = (
                    self.gemini_client
                    .interactions
                    .create(
                        model=self.gemini_model,
                        input=prompt,
                    )
                )

                text = (
                    interaction.output_text
                    or ""
                ).strip()

                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return text

            except Exception as exc:

                last_error = exc

                if attempt < self.gemini_retries:

                    wait_time = (
                        self.retry_delay
                        * attempt
                    )

                    print(
                        f"Gemini unavailable "
                        f"(attempt {attempt}/"
                        f"{self.gemini_retries})."
                    )

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)

        raise RuntimeError(
            f"Gemini failed: {last_error}"
        )

    # ========================================================
    # Groq
    # ========================================================

    def _call_groq(
        self,
        prompt: str,
    ) -> str:

        if self.groq_client is None:
            raise RuntimeError(
                "Groq is not configured."
            )

        response = (
            self.groq_client
            .chat
            .completions
            .create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                reasoning_effort="none",
                temperature=0,
                max_completion_tokens=600,
            )
        )

        text = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return text

    # ========================================================
    # Sources
    # ========================================================

    def _build_sources(
        self,
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        sources = []

        for number, item in enumerate(
            evidence,
            start=1,
        ):

            sources.append(
                {
                    "citation": number,
                    "filename": item.get(
                        "filename",
                        "Unknown source",
                    ),
                    "page_number": item.get(
                        "page_number",
                    ),
                    "category": item.get(
                        "category",
                        "unknown",
                    ),
                    "chunk_id": item.get(
                        "chunk_id",
                    ),
                }
            )

        return sources

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not evidence:

            return {
                "answer": (
                    "The available HSE documents do not "
                    "provide enough information to answer "
                    "this question reliably."
                ),
                "sources": [],
                "provider": None,
                "model": None,
                "grounded": False,
            }

        prompt = self._build_prompt(
            question,
            evidence,
        )

        gemini_error = None

        # ----------------------------------------------------
        # Gemini primary â€” only if enabled
        # ----------------------------------------------------

        if self.gemini_client is not None:

            print(
                f"Trying Gemini: "
                f"{self.gemini_model}"
            )

            try:

                answer = self._call_gemini(
                    prompt
                )

                return {
                    "answer": answer,
                    "sources": self._build_sources(
                        evidence
                    ),
                    "provider": "Gemini",
                    "model": self.gemini_model,
                    "grounded": True,
                }

            except Exception as exc:

                gemini_error = str(exc)

                print(
                    "Gemini unavailable."
                )

                if self.groq_client is not None:
                    print(
                        "Switching to Groq fallback..."
                    )

        # ----------------------------------------------------
        # Groq
        # ----------------------------------------------------

        if self.groq_client is not None:

            print(
                f"Using Groq: "
                f"{self.groq_model}"
            )

            try:

                answer = self._call_groq(
                    prompt
                )
                
                insufficient_message = (
                    "The available HSE documents do not provide "
                    "enough information to answer this question reliably."
                )

                if insufficient_message.lower() in answer.lower():
                    return {
                        "answer": insufficient_message,
                        "sources": [],
                        "provider": "Groq",
                        "model": self.groq_model,
                        "grounded": False,
                    }
                    
                result = {
                    "answer": answer,
                    "sources": self._build_sources(
                        evidence
                    ),
                    "provider": "Groq",
                    "model": self.groq_model,
                    "grounded": True,
                }

                if gemini_error:
                    result[
                        "primary_provider_error"
                    ] = gemini_error

                return result

            except Exception as exc:

                print("\n--- GROQ ERROR DEBUG ---")
                print("Type:", type(exc).__name__)
                print("Error:", str(exc))
                print(
                    "Underlying cause:",
                    repr(exc.__cause__),
                )
                print("------------------------\n")

                return {
                    "answer": (
                        "The HSE evidence was retrieved "
                        "successfully, but the language "
                        "model is temporarily unavailable."
                    ),
                    "sources": self._build_sources(
                        evidence
                    ),
                    "provider": "Groq",
                    "model": self.groq_model,
                    "grounded": False,
                    "error": str(exc),
                    "primary_provider_error": (
                        gemini_error
                    ),
                }

        return {
            "answer": (
                "The HSE evidence was retrieved "
                "successfully, but no language-model "
                "provider is currently available."
            ),
            "sources": self._build_sources(
                evidence
            ),
            "provider": None,
            "model": None,
            "grounded": False,
            "error": gemini_error,
        }

