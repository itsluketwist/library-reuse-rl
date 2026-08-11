"""Tests for the rerun helpers in benchmark/pipeline/build.py (loaded via pytest pythonpath)."""

from build import latest_records, succeeded_ids

from src.utils.io import save_jsonl


def _write(tmp_path, rows):
    path = tmp_path / "stage.jsonl"
    save_jsonl(rows, path)
    return path


_ROWS = [
    {"seed_id": "x", "kept": False, "drop_reason": "first try failed"},
    {"seed_id": "x", "kept": True},  # retried and succeeded
    {"seed_id": "y", "kept": False, "drop_reason": "still failing"},
]


class TestLatestRecords:
    """The latest record per seed wins, so retries override earlier attempts."""

    def test_last_wins(self, tmp_path) -> None:
        latest = latest_records(_write(tmp_path, _ROWS))
        assert latest["x"]["kept"] is True
        assert latest["y"]["kept"] is False

    def test_missing_file(self, tmp_path) -> None:
        assert latest_records(tmp_path / "none.jsonl") == {}


class TestSucceededIds:
    """Only seeds whose latest record succeeded are skipped; the rest are retried on a rerun."""

    def test_only_latest_kept(self, tmp_path) -> None:
        assert succeeded_ids(_write(tmp_path, _ROWS)) == {"x"}
