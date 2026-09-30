"""
SafetyCopilot — Gemini API Connection Test
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)


if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found in .env file."
    )


print("=" * 80)
print("  SAFETYCOPILOT — GEMINI API TEST")
print("=" * 80)

print(f"\nModel: {model_name}")
print("Connecting to Gemini...")


client = genai.Client(
    api_key=api_key
)


interaction = client.interactions.create(
    model=model_name,
    input=(
        "Reply with exactly this sentence and nothing else: "
        "SafetyCopilot Gemini connection successful."
    ),
)


response_text = interaction.output_text


print("\nGemini Response:")
print(response_text)


print("\n" + "=" * 80)

if (
    response_text
    and "SafetyCopilot Gemini connection successful"
    in response_text
):
    print("[PASS] Gemini API connection is working.")
else:
    print(
        "[WARNING] Gemini responded, "
        "but output differed from expected."
    )

print("=" * 80)