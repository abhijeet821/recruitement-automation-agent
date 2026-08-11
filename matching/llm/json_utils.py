"""
Getting reliable JSON out of a local LLM.

Hosted APIs have mature structured-output modes; local models are messier. Three
failure modes show up constantly and are handled here:

1. Reasoning models (qwen3 in particular) prepend a ``<think>…</think>`` block.
   We ask Ollama to disable thinking, but the tag still leaks on some models, so
   we strip it defensively rather than trusting the flag.
2. Models wrap output in ``` fences despite being told not to.
3. Models emit *almost* valid JSON — a trailing comma, a stray prose sentence
   before the object, smart quotes pasted from a resume.

``extract_json`` peels those layers in order and only then gives up. Doing this
in one audited place beats scattering ``try: json.loads`` across the codebase.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
# Control characters that are illegal inside JSON strings but routinely survive
# PDF text extraction and get echoed back by the model.
_BAD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_SMART_QUOTES = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", " ": " ",
}


class JSONExtractionError(ValueError):
    """Raised when no valid JSON value could be recovered from model output."""


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks, including one left unterminated by truncation."""
    text = _THINK_BLOCK.sub("", text)
    return _UNCLOSED_THINK.sub("", text)


def _normalise(text: str) -> str:
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    return _BAD_CONTROL.sub(" ", text)


def _find_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}`` / ``[...]`` span, quote-aware.

    A naive ``text[text.find('{'):text.rfind('}')+1]`` breaks the moment a
    resume contains a brace inside a string, which happens often enough with
    code snippets and LaTeX-formatted CVs to matter.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced (truncated generation): return the tail and let repair try.
    return text[start:]


def _repair(candidate: str) -> str:
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    # Close brackets left open by a truncated response.
    depth_curly = candidate.count("{") - candidate.count("}")
    depth_square = candidate.count("[") - candidate.count("]")
    if depth_square > 0:
        candidate += "]" * depth_square
    if depth_curly > 0:
        candidate += "}" * depth_curly
    return candidate


def extract_json(raw: str) -> Any:
    """Best-effort recovery of a JSON value from raw model output."""
    if not raw or not raw.strip():
        raise JSONExtractionError("model returned empty output")

    text = _normalise(strip_reasoning(raw)).strip()

    attempts: list[str] = [text]

    fenced = _CODE_FENCE.search(text)
    if fenced:
        attempts.append(fenced.group(1).strip())

    balanced = _find_balanced(text)
    if balanced:
        attempts.append(balanced.strip())
        attempts.append(_repair(balanced.strip()))

    for candidate in attempts:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    preview = text[:300].replace("\n", " ")
    raise JSONExtractionError(f"no valid JSON in model output (starts: {preview!r})")


def coerce_dict(value: Any) -> dict:
    """Normalise a parsed value to a dict.

    Models sometimes return ``[{...}]`` when asked for a single object; unwrap a
    one-element list rather than discarding an otherwise perfect extraction.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    raise JSONExtractionError(f"expected a JSON object, got {type(value).__name__}")


def coerce_list(value: Any, key: str | None = None) -> list:
    """Normalise a parsed value to a list, unwrapping ``{"key": [...]}``."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if key and isinstance(value.get(key), list):
            return value[key]
        for v in value.values():
            if isinstance(v, list):
                return v
    raise JSONExtractionError(f"expected a JSON array, got {type(value).__name__}")
