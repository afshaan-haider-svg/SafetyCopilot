"""
SafetyCopilot — Groq API Connection Test
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b",
)


if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env file."
    )


print("=" * 80)
print("  SAFETYCOPILOT — GROQ API TEST")
print("=" * 80)

print(f"\nModel: {model_name}")
print("Connecting to Groq...")


client = Groq(
    api_key=api_key
)


response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": (
                "Reply with exactly this sentence "
                "and nothing else: "
                "SafetyCopilot Groq connection successful."
            ),
        }
    ],
    reasoning_effort="none",
    temperature=0,
    max_completion_tokens=50,
)


response_text = (
    response.choices[0].message.content or ""
).strip()


print("\nGroq Response:")
print(response_text)


print("\n" + "=" * 80)

if (
    "SafetyCopilot Groq connection successful"
    in response_text
):
    print(
        "[PASS] Groq API connection is working."
    )
else:
    print(
        "[WARNING] Groq responded, "
        "but output differed from expected."
    )

print("=" * 80)