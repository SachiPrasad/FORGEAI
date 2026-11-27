"""Provider-agnostic LiteLLM structured completion helper."""

from __future__ import annotations

import json
import os
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_KNOWN_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_API_KEY",
    "COHERE_API_KEY",
)


def has_configured_llm_credentials(model: str | None = None) -> bool:
    """Return True if an API key for LiteLLM appears configured (or local Ollama is targeted)."""
    if model and model.startswith(("ollama/", "ollama_chat/")):
        return True
    return any(bool(os.getenv(key)) for key in _KNOWN_PROVIDER_ENV_KEYS)


def _extract_json_object(raw_text: str) -> str:
    """Extract a JSON object from raw LLM text, stripping Markdown code fences if present."""
    stripped = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]
    return stripped


def complete_structured(
    prompt: str,
    response_model: type[T],
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> T:
    """Call LiteLLM and parse the response into a Pydantic v2 model."""
    from litellm import completion

    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    system_msg = (
        "You must respond strictly with a single valid JSON object conforming to this JSON Schema:\n"
        f"{schema_json}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_model,
        )
    except Exception:
        # Fallback for providers/models that do not accept Pydantic response_format directly
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    content = response.choices[0].message.content or ""
    json_payload = _extract_json_object(content)
    return response_model.model_validate_json(json_payload)

