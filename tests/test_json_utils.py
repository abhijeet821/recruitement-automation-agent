"""Recovering JSON from messy local-model output."""

from __future__ import annotations

import pytest

from matching.llm.json_utils import (
    JSONExtractionError,
    coerce_dict,
    coerce_list,
    extract_json,
    strip_reasoning,
)


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_strips_think_block():
    raw = '<think>Let me consider this carefully.</think>{"a": 1}'
    assert extract_json(raw) == {"a": 1}


def test_strips_unterminated_think_block():
    """A truncated generation can leave <think> open with no closing tag."""
    raw = '{"a": 1}<think>and now I will keep reasoning forever'
    assert extract_json(raw) == {"a": 1}


def test_strips_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_ignores_prose_preamble():
    assert extract_json('Sure! Here is the result:\n{"a": 1}') == {"a": 1}


def test_repairs_trailing_comma():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_closes_truncated_object():
    assert extract_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1, 2]}


def test_brace_inside_string_does_not_break_parsing():
    """A naive first-brace-to-last-brace slice mishandles this."""
    raw = '{"code": "if (x) { return {}; }", "n": 2}'
    assert extract_json(raw)["n"] == 2


def test_normalises_smart_quotes_outside_strings():
    assert extract_json('{"a": 1} ') == {"a": 1}


def test_strips_control_characters():
    assert extract_json('{"a": "line\x0bbreak"}')["a"] == "line break"


def test_empty_output_raises():
    with pytest.raises(JSONExtractionError):
        extract_json("")


def test_garbage_raises():
    with pytest.raises(JSONExtractionError):
        extract_json("I am afraid I cannot help with that request.")


def test_coerce_dict_unwraps_single_element_list():
    assert coerce_dict([{"a": 1}]) == {"a": 1}


def test_coerce_dict_rejects_scalar():
    with pytest.raises(JSONExtractionError):
        coerce_dict(42)


def test_coerce_list_unwraps_keyed_object():
    assert coerce_list({"verdicts": [1, 2]}, "verdicts") == [1, 2]


def test_strip_reasoning_is_case_insensitive():
    assert strip_reasoning("<THINK>x</THINK>ok").strip() == "ok"
