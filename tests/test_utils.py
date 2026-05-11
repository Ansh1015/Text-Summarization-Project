import tempfile
from pathlib import Path

import pytest
import yaml

from textSummarizer.utils.common import create_directories, get_size, read_yaml


def _write_yaml(content: dict, path: Path):
    with open(path, "w") as f:
        yaml.dump(content, f)


class TestReadYaml:
    def test_reads_valid_yaml(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        _write_yaml({"key": "value", "nested": {"a": 1}}, yaml_path)
        result = read_yaml(yaml_path)
        assert result.key == "value"
        assert result.nested.a == 1

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_yaml(tmp_path / "nonexistent.yaml")

    def test_raises_on_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_yaml(yaml_path)


class TestCreateDirectories:
    def test_creates_single_directory(self, tmp_path):
        new_dir = tmp_path / "subdir"
        create_directories([new_dir])
        assert new_dir.is_dir()

    def test_creates_multiple_directories(self, tmp_path):
        dirs = [tmp_path / "a", tmp_path / "b" / "c"]
        create_directories(dirs)
        for d in dirs:
            assert d.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path):
        existing = tmp_path / "exists"
        existing.mkdir()
        create_directories([existing])  # Should not raise


class TestGetSize:
    def test_returns_size_string(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 2048)
        result = get_size(f)
        assert "KB" in result
        assert "~" in result
