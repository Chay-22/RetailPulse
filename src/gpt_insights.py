"""
RetailPulse - Optional GPT Insights

Provides OpenAI-powered retail insights when an API key is configured.
RetailPulse continues to work without OpenAI, using the application's
non-GPT/local insight mechanisms.
"""

import logging
import os
from typing import Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


def _get_openai_client() -> Optional[OpenAI]:
    """
    Create an OpenAI client only when an API key is available.

    Returns:
        OpenAI client if OPENAI_API_KEY is configured, otherwise None.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        logger.warning("Unable to initialize OpenAI client: %s", exc)
        return None


def generate_gpt_insights(context_text: str) -> str:
    """
    Generate retail insights using OpenAI when configured.

    RetailPulse remains functional without an OpenAI API key.

    Args:
        context_text: Retail analytics context used to generate insights.

    Returns:
        Generated insights, or a safe fallback message when GPT
        functionality is unavailable.
    """
    if not context_text or not str(context_text).strip():
        return "No sufficient data is available to generate GPT insights."

    client = _get_openai_client()

    if client is None:
        return (
            "GPT insights are unavailable because an OpenAI API key "
            "is not configured. RetailPulse analytics remain available."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a retail business analyst.",
                },
                {
                    "role": "user",
                    "content": (
                        "Give concise, actionable retail insights and "
                        f"recommendations based on this data:\n{context_text}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content

        if content:
            return content.strip()

        return "GPT returned no insights."

    except Exception as exc:
        logger.warning("GPT insight generation failed: %s", exc)
        return "GPT insights are currently unavailable."