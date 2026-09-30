"""
SafetyCopilot — API Integration Tests

Tests:
1. Root endpoint
2. Health endpoint
3. Empty question validation
4. Valid HSE question
5. Out-of-domain rejection
6. Conversation session and memory
"""

import uuid

import requests


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120


# ============================================================
# Helper
# ============================================================

def ask_api(question: str, session_id: str) -> requests.Response:
    """
    Send a question to the SafetyCopilot /ask endpoint.
    """

    return requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": question,
            "session_id": session_id,
        },
        timeout=TIMEOUT,
    )


# ============================================================
# Test 1 — Root endpoint
# ============================================================

def test_root_endpoint():
    response = requests.get(
        f"{BASE_URL}/",
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "SafetyCopilot API"
    assert data["status"] == "running"


# ============================================================
# Test 2 — Health endpoint
# ============================================================

def test_health_endpoint():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=TIMEOUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "SafetyCopilot"

    assert "documents" in data
    assert "chunks" in data

    assert data["documents"] > 0
    assert data["chunks"] > 0


# ============================================================
# Test 3 — Empty question validation
# ============================================================

def test_empty_question_rejected():
    session_id = (
        "pytest-empty-"
        + uuid.uuid4().hex[:8]
    )

    response = ask_api(
        question="   ",
        session_id=session_id,
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
    assert data["detail"] == "Question cannot be empty."


# ============================================================
# Test 4 — Valid HSE question
# ============================================================

def test_valid_hse_question():
    session_id = (
        "pytest-hse-"
        + uuid.uuid4().hex[:8]
    )

    response = ask_api(
        question=(
            "What precautions should be taken "
            "when working at height?"
        ),
        session_id=session_id,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What precautions should be taken "
        "when working at height?"
    )

    assert data["session_id"] == session_id

    assert data["grounded"] is True

    assert data["answer"].strip()

    assert len(data["sources"]) > 0

    categories = {
        source["category"]
        for source in data["sources"]
    }

    assert "working_at_height" in categories

    assert data["provider"] is not None
    assert data["model"] is not None


# ============================================================
# Test 5 — Out-of-domain rejection
# ============================================================

def test_out_of_domain_question():
    session_id = (
        "pytest-ood-"
        + uuid.uuid4().hex[:8]
    )

    response = ask_api(
        question=(
            "Who is the current president "
            "of the United States?"
        ),
        session_id=session_id,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["session_id"] == session_id

    assert data["grounded"] is False

    assert data["sources"] == []

    assert data["answer"].strip()


# ============================================================
# Test 6 — Conversation memory
# ============================================================

def test_conversation_memory():
    session_id = (
        "pytest-memory-"
        + uuid.uuid4().hex[:8]
    )

    # --------------------------------------------------------
    # Turn 1
    # --------------------------------------------------------

    first_response = ask_api(
        question=(
            "What are the main dangers of "
            "working inside a confined space?"
        ),
        session_id=session_id,
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert first_data["session_id"] == session_id
    assert first_data["grounded"] is True
    assert first_data["answer"].strip()

    first_categories = {
        source["category"]
        for source in first_data["sources"]
    }

    assert "confined_space" in first_categories

    # --------------------------------------------------------
    # Turn 2
    #
    # This question intentionally does NOT mention
    # "confined space".
    # --------------------------------------------------------

    second_response = ask_api(
        question="How can these risks be prevented?",
        session_id=session_id,
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert second_data["session_id"] == session_id
    assert second_data["grounded"] is True
    assert second_data["answer"].strip()

    second_categories = {
        source["category"]
        for source in second_data["sources"]
    }

    assert "confined_space" in second_categories


# ============================================================
# Test 7 — Different sessions remain independent
# ============================================================

def test_different_sessions_are_independent():
    session_one = (
        "pytest-session-a-"
        + uuid.uuid4().hex[:8]
    )

    session_two = (
        "pytest-session-b-"
        + uuid.uuid4().hex[:8]
    )

    response_one = ask_api(
        question=(
            "What personal protective equipment "
            "should workers use?"
        ),
        session_id=session_one,
    )

    response_two = ask_api(
        question=(
            "What precautions should be taken "
            "when working at height?"
        ),
        session_id=session_two,
    )

    assert response_one.status_code == 200
    assert response_two.status_code == 200

    data_one = response_one.json()
    data_two = response_two.json()

    assert data_one["session_id"] == session_one
    assert data_two["session_id"] == session_two

    assert session_one != session_two