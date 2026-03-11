from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBMIT_PATH = REPO_ROOT / "dco_qwen3_visualize" / "submit.py"
spec = importlib.util.spec_from_file_location("dco_qwen3_visualize_submit", SUBMIT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

_extract_balanced_literal = module._extract_balanced_literal
_collapse_newlines_inside_strings = module._collapse_newlines_inside_strings
_strip_ansi = module._strip_ansi


def test_strip_ansi_removes_terminal_sequences() -> None:
    assert _strip_ansi("\x1b[1mhello\x1b[0m") == "hello"


def test_extract_balanced_literal_for_dict() -> None:
    payload = "Inputs\n{'a': {'b': 1}, 'c': [1, 2, 3]}\nOutputs\nNone"
    assert _extract_balanced_literal(payload, "{") == "{'a': {'b': 1}, 'c': [1, 2, 3]}"


def test_extract_balanced_literal_for_list() -> None:
    payload = "prefix\n[{'id': {'name': 'a0'}}, {'id': {'name': 'b1'}}]\nsuffix"
    assert _extract_balanced_literal(payload, "[") == "[{'id': {'name': 'a0'}}, {'id': {'name': 'b1'}}]"


def test_collapse_newlines_inside_strings_preserves_literal() -> None:
    payload = "{'path': 's3://bucket/very/long/\nkey/file.parquet', 'n': 1}"
    assert _collapse_newlines_inside_strings(payload) == "{'path': 's3://bucket/very/long/key/file.parquet', 'n': 1}"
