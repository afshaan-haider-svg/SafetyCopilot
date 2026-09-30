"""
SafetyCopilot — Persistent Conversation Memory

Provides SQLite-backed session history for
multi-turn SafetyCopilot conversations.

Conversation history survives FastAPI server restarts.

The retrieval-context logic distinguishes between:

1. Follow-up questions that depend on previous context.
2. Standalone questions that introduce a new topic.

For follow-up questions, retrieval uses a compact
topic + current-intent query instead of injecting the
entire previous answer into retrieval.
"""

import re
import sqlite3

from pathlib import Path
from threading import Lock
from typing import Dict, List


class ConversationMemory:
    """
    Store recent conversation turns separately
    for each session_id using SQLite.

    Also determine whether the current question
    depends on previous conversation context.
    """

    def __init__(
        self,
        max_turns: int = 5,
        db_path: str | Path | None = None,
    ):
        if max_turns < 1:
            raise ValueError(
                "max_turns must be at least 1."
            )

        self.max_turns = max_turns

        # -----------------------------------------
        # Database location
        # -----------------------------------------

        if db_path is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parent
                .parent
                .parent
            )

            db_path = (
                project_root
                / "data"
                / "memory"
                / "conversation_memory.db"
            )

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = Lock()

        self._initialize_database()

    # ---------------------------------------------
    # Database connection
    # ---------------------------------------------

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """
        Create and return a SQLite connection.
        """

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        return connection

    # ---------------------------------------------
    # Initialize database
    # ---------------------------------------------

    def _initialize_database(
        self,
    ) -> None:
        """
        Create conversation table and indexes
        if they do not already exist.
        """

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversation_session
                ON conversation_turns(session_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversation_session_id
                ON conversation_turns(session_id, id)
                """
            )

            connection.commit()

    # ---------------------------------------------
    # Add turn
    # ---------------------------------------------

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Add one user/assistant turn to a session.

        Only the newest max_turns are retained.
        """

        session_id = session_id.strip()
        question = question.strip()
        answer = answer.strip()

        if not session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        if not question:
            return

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO conversation_turns (
                        session_id,
                        question,
                        answer
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        session_id,
                        question,
                        answer,
                    ),
                )

                connection.execute(
                    """
                    DELETE FROM conversation_turns
                    WHERE session_id = ?
                    AND id NOT IN (
                        SELECT id
                        FROM conversation_turns
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    (
                        session_id,
                        session_id,
                        self.max_turns,
                    ),
                )

                connection.commit()

    # ---------------------------------------------
    # Get history
    # ---------------------------------------------

    def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:
        """
        Return conversation history from
        oldest to newest.
        """

        session_id = session_id.strip()

        if not session_id:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    question,
                    answer
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [
            {
                "question": row["question"],
                "answer": row["answer"],
            }
            for row in rows
        ]

    # ---------------------------------------------
    # Follow-up detection
    # ---------------------------------------------

    def is_follow_up(
        self,
        question: str,
    ) -> bool:
        """
        Determine whether a question likely
        depends on previous conversation context.
        """

        question = question.strip()

        if not question:
            return False

        normalized = re.sub(
            r"\s+",
            " ",
            question.lower(),
        ).strip()

        # Explicit contextual references.
        contextual_reference_patterns = [
            r"\bthis\b",
            r"\bthat\b",
            r"\bthese\b",
            r"\bthose\b",
            r"\bit\b",
            r"\bits\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\btheir\b",
            r"\bsuch\b",
            r"\babove\b",
            r"\bprevious\b",
            r"\bearlier\b",
            r"\bsame\b",
            r"\bformer\b",
            r"\blatter\b",
            r"\bmentioned\b",
            r"\blisted\b",
        ]

        for pattern in contextual_reference_patterns:
            if re.search(
                pattern,
                normalized,
            ):
                return True

        # Common follow-up openings.
        follow_up_openings = (
            "what about ",
            "how about ",
            "and what ",
            "and how ",
            "and why ",
            "and which ",
            "also ",
            "then ",
            "so ",
            "why ",
            "how ",
            "which ones",
            "which one",
            "what else",
            "anything else",
            "explain further",
            "explain more",
            "tell me more",
            "can you explain",
            "could you explain",
            "what does that mean",
            "what does this mean",
            "can they",
            "can it",
            "are they",
            "is it",
        )

        if normalized.startswith(
            follow_up_openings
        ):
            word_count = len(
                normalized.split()
            )

            # Short how/why questions are likely
            # contextual follow-ups.
            if normalized.startswith(
                (
                    "how ",
                    "why ",
                )
            ):
                if word_count <= 8:
                    return True

            else:
                return True

        short_follow_ups = {
            "why?",
            "why",
            "how?",
            "how",
            "what next?",
            "what next",
            "then what?",
            "then what",
            "what else?",
            "what else",
            "and?",
            "and",
        }

        if normalized in short_follow_ups:
            return True

        return False

    # ---------------------------------------------
    # Build retrieval context
    # ---------------------------------------------

    def build_context(
        self,
        session_id: str,
        current_question: str,
        history_turns: int = 2,
    ) -> str:
        """
        Build a retrieval-friendly query.

        Standalone questions are returned unchanged.

        For context-dependent follow-ups, use the
        previous user question as a topic anchor and
        preserve the current follow-up as the new intent.

        IMPORTANT:
        The previous assistant answer is deliberately
        NOT inserted into the retrieval query because
        a long previous answer can dominate retrieval
        and cause the retriever to repeat old evidence
        instead of finding evidence for the new intent.
        """

        current_question = (
            current_question.strip()
        )

        if not current_question:
            return ""

        if history_turns < 1:
            return current_question

        history = self.get_history(
            session_id
        )

        if not history:
            return current_question

        # Standalone/new-topic questions should
        # remain completely independent.
        if not self.is_follow_up(
            current_question
        ):
            return current_question

        recent_history = history[
            -history_turns:
        ]

        # Find the most recent previous question.
        previous_question = ""

        for turn in reversed(
            recent_history
        ):
            candidate = (
                turn.get(
                    "question",
                    "",
                ).strip()
            )

            if candidate:
                previous_question = candidate
                break

        if not previous_question:
            return current_question

        normalized_question = re.sub(
            r"\s+",
            " ",
            current_question,
        ).strip()

        # -----------------------------------------
        # Compact contextual retrieval query
        # -----------------------------------------
        #
        # Example:
        #
        # Previous:
        # What are the main dangers of working
        # inside a confined space?
        #
        # Current:
        # How can these risks be prevented?
        #
        # Query:
        #
        # Topic: What are the main dangers of
        # working inside a confined space?
        #
        # Follow-up intent:
        # How can these risks be prevented?
        #
        # This keeps the subject and the NEW intent
        # without flooding retrieval with the
        # previous generated answer.

        retrieval_query = (
            f"Topic: {previous_question}\n"
            f"Follow-up intent: "
            f"{normalized_question}"
        )

        return retrieval_query

    # ---------------------------------------------
    # Format history for optional LLM use
    # ---------------------------------------------

    def format_history(
        self,
        session_id: str,
        history_turns: int = 2,
    ) -> str:
        """
        Format recent conversation turns for
        optional use inside a generation prompt.

        This is separate from build_context().
        """

        if history_turns < 1:
            return ""

        history = self.get_history(
            session_id
        )

        if not history:
            return ""

        recent_history = history[
            -history_turns:
        ]

        formatted = []

        for turn in recent_history:
            formatted.append(
                f"User: {turn['question']}"
            )

            formatted.append(
                "SafetyCopilot: "
                f"{turn['answer']}"
            )

        return "\n".join(
            formatted
        )

    # ---------------------------------------------
    # Clear one session
    # ---------------------------------------------

    def clear_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete all turns belonging to one session.

        Returns True if rows were deleted.
        """

        session_id = session_id.strip()

        if not session_id:
            return False

        with self._lock:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM conversation_turns
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )

                connection.commit()

                return cursor.rowcount > 0

    # ---------------------------------------------
    # Clear all sessions
    # ---------------------------------------------

    def clear_all(
        self,
    ) -> None:
        """
        Delete all stored conversation turns.
        """

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    DELETE FROM conversation_turns
                    """
                )

                connection.commit()

    # ---------------------------------------------
    # Session exists
    # ---------------------------------------------

    def session_exists(
        self,
        session_id: str,
    ) -> bool:
        """
        Check whether a session exists.
        """

        session_id = session_id.strip()

        if not session_id:
            return False

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM conversation_turns
                WHERE session_id = ?
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        return row is not None

    # ---------------------------------------------
    # Turn count
    # ---------------------------------------------

    def turn_count(
        self,
        session_id: str,
    ) -> int:
        """
        Return the number of stored turns
        for one session.
        """

        session_id = session_id.strip()

        if not session_id:
            return 0

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM conversation_turns
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return int(
            row["count"]
        )

    # ---------------------------------------------
    # List sessions
    # ---------------------------------------------

    def list_sessions(
        self,
    ) -> List[str]:
        """
        Return all stored session IDs.
        """

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT session_id
                FROM conversation_turns
                ORDER BY session_id ASC
                """
            ).fetchall()

        return [
            row["session_id"]
            for row in rows
        ]

    # ---------------------------------------------
    # Total turns
    # ---------------------------------------------

    def total_turns(
        self,
    ) -> int:
        """
        Return the total number of conversation
        turns stored across all sessions.
        """

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM conversation_turns
                """
            ).fetchone()

        return int(
            row["count"]
        )