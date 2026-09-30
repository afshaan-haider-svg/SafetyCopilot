"""
SafetyCopilot — Automated RAG Evaluation

Tests:
1. Confined-space retrieval and generation
2. PPE retrieval and generation
3. Working-at-height retrieval and generation
4. Multi-source hazardous-substances query
5. Out-of-domain rejection
6. Multi-turn conversation memory
"""

import json
import urllib.request
import urllib.error
import uuid
import time


API_URL = "http://127.0.0.1:8000/ask"

# Delay between API requests to reduce consecutive LLM/API failures
REQUEST_DELAY = 3


TEST_CASES = [
    {
        "name": "Confined Space",
        "question": "What are the main dangers of working inside a confined space?",
        "expected_categories": ["confined_space"],
        "should_be_grounded": True,
    },
    {
        "name": "PPE",
        "question": "What personal protective equipment should workers use?",
        "expected_categories": ["ppe"],
        "should_be_grounded": True,
    },
    {
        "name": "Working at Height",
        "question": "What precautions should be taken when working at height?",
        "expected_categories": ["working_at_height"],
        "should_be_grounded": True,
    },
    {
        "name": "Hazardous Substances",
        "question": (
            "What are the main risks and precautions "
            "when working with hazardous substances?"
        ),
        "expected_categories": [
            "ppe",
            "electrical_safety",
            "confined_space",
        ],
        "should_be_grounded": True,
    },
    {
        "name": "Out of Domain",
        "question": "Who is the current president of the United States?",
        "expected_categories": [],
        "should_be_grounded": False,
    },
]


def ask_api(
    question: str,
    session_id: str = "evaluation-default",
) -> dict:
    """
    Send one question to the SafetyCopilot API.
    """

    payload = json.dumps(
        {
            "question": question,
            "session_id": session_id,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"API returned HTTP {exc.code}: "
            f"{exc.read().decode('utf-8')}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to SafetyCopilot API. "
            "Make sure Uvicorn is running."
        ) from exc


def evaluate_case(test_case: dict) -> dict:
    """
    Evaluate one independent RAG test case.
    """

    # Give every independent test its own session.
    session_id = (
        "eval-"
        + test_case["name"]
        .lower()
        .replace(" ", "-")
        + "-"
        + uuid.uuid4().hex[:8]
    )

    result = ask_api(
        question=test_case["question"],
        session_id=session_id,
    )

    # Prevent rapid consecutive requests to the LLM provider.
    time.sleep(REQUEST_DELAY)

    sources = result.get("sources", [])

    returned_categories = {
        source.get("category")
        for source in sources
        if source.get("category")
    }

    expected_categories = set(
        test_case["expected_categories"]
    )

    grounded = result.get(
        "grounded",
        False,
    )

    grounding_pass = (
        grounded
        == test_case["should_be_grounded"]
    )

    if test_case["should_be_grounded"]:

        source_pass = bool(
            returned_categories
            & expected_categories
        )

        answer_pass = bool(
            result.get(
                "answer",
                "",
            ).strip()
        )

    else:

        source_pass = (
            len(sources) == 0
        )

        answer_pass = (
            grounded is False
        )

    passed = (
        grounding_pass
        and source_pass
        and answer_pass
    )

    return {
        "name": test_case["name"],
        "passed": passed,
        "grounding_pass": grounding_pass,
        "source_pass": source_pass,
        "answer_pass": answer_pass,
        "grounded": grounded,
        "returned_categories": sorted(
            returned_categories
        ),
        "answer": result.get(
            "answer",
            "",
        ),
    }


def evaluate_conversation_memory() -> dict:
    """
    Evaluate multi-turn conversation memory.

    Turn 2 deliberately does not mention
    'confined space'.

    The system must use Turn 1 context to understand
    what 'these risks' refers to.
    """

    session_id = (
        "memory-eval-"
        + uuid.uuid4().hex[:8]
    )

    first_question = (
        "What are the main dangers of "
        "working inside a confined space?"
    )

    follow_up_question = (
        "How can these risks be prevented?"
    )

    # -------------------------
    # Turn 1
    # -------------------------

    first_result = ask_api(
        question=first_question,
        session_id=session_id,
    )

    # Wait before sending the follow-up request.
    time.sleep(REQUEST_DELAY)

    first_sources = first_result.get(
        "sources",
        [],
    )

    first_categories = {
        source.get("category")
        for source in first_sources
        if source.get("category")
    }

    first_turn_pass = (
        first_result.get("grounded") is True
        and "confined_space"
        in first_categories
    )

    # -------------------------
    # Turn 2
    # -------------------------

    second_result = ask_api(
        question=follow_up_question,
        session_id=session_id,
    )

    # Small cooldown after the final LLM request.
    time.sleep(REQUEST_DELAY)

    second_sources = second_result.get(
        "sources",
        [],
    )

    second_categories = {
        source.get("category")
        for source in second_sources
        if source.get("category")
    }

    session_pass = (
        second_result.get("session_id")
        == session_id
    )

    grounding_pass = (
        second_result.get("grounded")
        is True
    )

    source_pass = (
        "confined_space"
        in second_categories
    )

    answer_pass = bool(
        second_result.get(
            "answer",
            "",
        ).strip()
    )

    passed = (
        first_turn_pass
        and session_pass
        and grounding_pass
        and source_pass
        and answer_pass
    )

    return {
        "name": "Conversation Memory",
        "passed": passed,
        "first_turn_pass": first_turn_pass,
        "session_pass": session_pass,
        "grounding_pass": grounding_pass,
        "source_pass": source_pass,
        "answer_pass": answer_pass,
        "grounded": second_result.get(
            "grounded",
            False,
        ),
        "returned_categories": sorted(
            second_categories
        ),
        "answer": second_result.get(
            "answer",
            "",
        ),
        "session_id": session_id,
    }


def print_result(
    number: int,
    total: int,
    result: dict,
) -> None:
    """
    Print one evaluation result.
    """

    print(
        f"\n[{number}/{total}] "
        f"{result['name']}"
    )

    status = (
        "PASS"
        if result.get("passed")
        else "FAIL"
    )

    print(
        f"Result     : {status}"
    )

    print(
        f"Grounded   : "
        f"{result.get('grounded', False)}"
    )

    print(
        "Categories : "
        f"{result.get('returned_categories', [])}"
    )

    if result["name"] == "Conversation Memory":

        print(
            f"Session    : "
            f"{result.get('session_id')}"
        )

        print(
            f"Turn 1     : "
            f"{'PASS' if result.get('first_turn_pass') else 'FAIL'}"
        )

        print(
            f"Session ID : "
            f"{'PASS' if result.get('session_pass') else 'FAIL'}"
        )

        print(
            f"Follow-up  : "
            f"{'PASS' if result.get('source_pass') else 'FAIL'}"
        )

    answer = result.get(
        "answer",
        "",
    )

    if answer:
        print(
            f"Answer     : "
            f"{answer[:250]}..."
        )


def main():

    print("=" * 80)
    print(
        "SAFETYCOPILOT — AUTOMATED RAG EVALUATION"
    )
    print("=" * 80)

    results = []

    total_tests = (
        len(TEST_CASES) + 1
    )

    # -------------------------------------------------
    # Standard RAG tests
    # -------------------------------------------------

    for number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            f"\nRunning test: "
            f"{test_case['name']}"
        )

        print(
            f"Question: "
            f"{test_case['question']}"
        )

        try:

            result = evaluate_case(
                test_case
            )

        except Exception as exc:

            result = {
                "name": test_case["name"],
                "passed": False,
                "error": str(exc),
            }

            print(
                f"ERROR: {exc}"
            )

        results.append(result)

        print_result(
            number,
            total_tests,
            result,
        )

    # -------------------------------------------------
    # Conversation memory test
    # -------------------------------------------------

    print(
        "\nRunning multi-turn "
        "conversation memory test..."
    )

    try:

        memory_result = (
            evaluate_conversation_memory()
        )

    except Exception as exc:

        memory_result = {
            "name": "Conversation Memory",
            "passed": False,
            "error": str(exc),
        }

        print(
            f"ERROR: {exc}"
        )

    results.append(
        memory_result
    )

    print_result(
        total_tests,
        total_tests,
        memory_result,
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    passed_count = sum(
        1
        for result in results
        if result.get("passed")
    )

    total = len(results)

    failed_count = (
        total - passed_count
    )

    score = (
        passed_count
        / total
        * 100
        if total
        else 0
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Passed : "
        f"{passed_count}/{total}"
    )

    print(
        f"Failed : "
        f"{failed_count}/{total}"
    )

    print(
        f"Score  : "
        f"{score:.1f}%"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()