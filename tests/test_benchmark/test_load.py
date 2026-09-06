"""Tests for libreusebench/load.py's load_records: the shared json/jsonl loading primitive."""

import json

import pytest
from libreusebench.load import load_records

from src.utils.io import save_jsonl


_RECORDS = [
    {"task_id": "LRB/T000", "response": "x"},
    {"task_id": "LRB/T001", "response": "y"},
]


class TestLoadRecords:
    """Records load identically whether the file is .json (a list) or .jsonl (one per line)."""

    def test_jsonl_file(self, tmp_path) -> None:
        path = tmp_path / "records.jsonl"
        save_jsonl(_RECORDS, path)
        assert load_records(str(path)) == _RECORDS

    def test_json_file(self, tmp_path) -> None:
        path = tmp_path / "records.json"
        path.write_text(json.dumps(_RECORDS))
        assert load_records(str(path)) == _RECORDS

    def test_unsupported_extension_errors(self, tmp_path) -> None:
        path = tmp_path / "records.txt"
        path.write_text(json.dumps(_RECORDS))
        with pytest.raises(ValueError):
            load_records(str(path))
