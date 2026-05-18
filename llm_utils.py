"""
Local LLM utilities: OpenAI client, completion wrapper, JSON-output parser.

Standalone — no external private dependencies. Reads the OpenAI API key from
either OPENAI_API_KEY env var or a local `.openai_key` file (gitignored).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_KEY_FILE = HERE / ".openai_key"


def openai_client():
    """Resolve the OpenAI key and return an authenticated client.

    Resolution order:
      1. OPENAI_API_KEY env var
      2. ./.openai_key file in this directory (gitignored)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and _KEY_FILE.exists():
        content = _KEY_FILE.read_text(encoding="utf-8").strip()
        if content.startswith("OPENAI_API_KEY="):
            content = content.split("=", 1)[1].strip().strip("'\"")
        if content.startswith("sk-"):
            api_key = content
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not set. Either set the env var, or paste your "
            f"key into {_KEY_FILE} (one line, starts with sk-)."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def llm_call(system: str, user: str, *,
             max_tokens: int = 2500,
             model: str = "gpt-4o",
             temperature: float = 0.0,
             json_mode: bool = True) -> str:
    """Single-shot completion. Defaults to JSON-mode for structured output.

    `model` options: "gpt-4o" (quality), "gpt-4o-mini" (4x cheaper + faster).
    """
    client = openai_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def parse_json_from_response(text: str) -> dict:
    """Extract a JSON object from an LLM response. Handles ```json fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"No JSON in response: {text[:300]}")
    return json.loads(text[start: end + 1])
