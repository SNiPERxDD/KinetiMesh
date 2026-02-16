"""
Tests for the MeshIngestor (src/ingestor/scanner.py).

Covers:
    - Basic scan of a valid repo
    - Incremental scan (only changed files re-yielded)
    - Empty directory scan
    - Pruned directories are skipped (.git, node_modules, __pycache__)
    - Binary / skip-extension files excluded
    - Single-file scan
    - Deleted file detection
    - CodeFile dataclass field correctness
    - Edge cases: deeply nested, unicode filenames, large files
"""

from pathlib import Path

import pytest

from src.ingestor.scanner import (
    MeshIngestor,
    CodeFile,
    EXTENSION_TO_LANGUAGE,
    PRUNE_DIRS,
    MAX_FILE_BYTES,
)


class TestMeshIngestorScan:
    """Tests for the full scan() method."""

    def test_scan_yields_expected_files(self, tmp_repo: Path):
        """Scan should yield all non-pruned, non-binary source files."""
        ingestor = MeshIngestor(str(tmp_repo))
        files = list(ingestor.scan(incremental=False))

        rel_paths = {f.rel_path for f in files}
        assert "main.py" in rel_paths, "Expected main.py in scan results"
        assert "src/auth.js" in rel_paths, "Expected src/auth.js in scan results"
        assert "src/server.ts" in rel_paths, "Expected src/server.ts in scan results"
        assert "docs/notes.md" in rel_paths, "Expected docs/notes.md in scan results"

    def test_scan_empty_file_included(self, tmp_repo: Path):
        """Empty .py files should still be yielded (content may be empty)."""
        ingestor = MeshIngestor(str(tmp_repo))
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}
        # empty.py has no content but is a valid python file
        assert "empty.py" in rel_paths, "Empty .py file should be included"

    def test_scan_empty_directory(self, empty_dir: Path):
        """Scanning an empty directory should yield no files."""
        ingestor = MeshIngestor(str(empty_dir))
        files = list(ingestor.scan(incremental=False))
        assert files == [], f"Expected no files, got {len(files)}"

    def test_scan_nonexistent_directory(self, tmp_path: Path):
        """Scanning a nonexistent directory should yield no files (no crash)."""
        ingestor = MeshIngestor(str(tmp_path / "does_not_exist"))
        files = list(ingestor.scan(incremental=False))
        assert files == [], "Nonexistent directory should yield empty results"

    def test_scan_pruned_directories_skipped(self, nested_repo: Path):
        """Files inside .git, node_modules, __pycache__ must NOT appear."""
        ingestor = MeshIngestor(str(nested_repo))
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}

        for pruned in [".git/should_skip.py", "node_modules/should_skip.py",
                       "__pycache__/should_skip.py"]:
            assert pruned not in rel_paths, f"{pruned} should be pruned"

    def test_scan_deeply_nested_files_found(self, nested_repo: Path):
        """Deeply nested files should be found."""
        ingestor = MeshIngestor(str(nested_repo))
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}
        assert "a/b/c/d/deep.py" in rel_paths, "Deep nested file should be found"
        assert "valid.py" in rel_paths, "Top-level valid file should be found"

    def test_scan_skip_binary_extensions(self, tmp_repo: Path):
        """Files with binary extensions (.zip, .png, etc.) must be excluded."""
        (tmp_repo / "archive.zip").write_bytes(b"\x00" * 100)
        (tmp_repo / "image.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_repo / "lib.so").write_bytes(b"\x00" * 50)

        ingestor = MeshIngestor(str(tmp_repo))
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}

        assert "archive.zip" not in rel_paths
        assert "image.png" not in rel_paths
        assert "lib.so" not in rel_paths

    def test_scan_skip_oversized_files(self, tmp_repo: Path):
        """Files exceeding MAX_FILE_BYTES should be excluded."""
        big_file = tmp_repo / "huge.py"
        big_file.write_text("x = 1\n" * (MAX_FILE_BYTES // 5), encoding="utf-8")
        assert big_file.stat().st_size > MAX_FILE_BYTES

        ingestor = MeshIngestor(str(tmp_repo))
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}
        assert "huge.py" not in rel_paths, "Oversized file should be skipped"


class TestIncrementalScan:
    """Tests for incremental (hash-based change detection) scanning."""

    def test_incremental_first_scan_yields_all(self, tmp_repo: Path):
        """First incremental scan should yield all files (nothing in state)."""
        ingestor = MeshIngestor(str(tmp_repo))
        first = list(ingestor.scan(incremental=True))
        assert len(first) > 0, "First scan should yield files"

    def test_incremental_second_scan_yields_nothing(self, tmp_repo: Path):
        """Second scan without changes should yield zero files."""
        ingestor = MeshIngestor(str(tmp_repo))
        list(ingestor.scan(incremental=True))  # populate state
        second = list(ingestor.scan(incremental=True))
        assert second == [], "No files changed, should yield nothing"

    def test_incremental_detects_modified_file(self, tmp_repo: Path):
        """Modifying a file should cause it to be re-yielded."""
        ingestor = MeshIngestor(str(tmp_repo))
        list(ingestor.scan(incremental=True))  # first scan

        # Modify main.py
        (tmp_repo / "main.py").write_text("# modified\nprint('changed')\n", encoding="utf-8")
        second = list(ingestor.scan(incremental=True))

        assert len(second) == 1, f"Expected 1 modified file, got {len(second)}"
        assert second[0].rel_path == "main.py"

    def test_incremental_detects_new_file(self, tmp_repo: Path):
        """Adding a new file should cause it to appear in next scan."""
        ingestor = MeshIngestor(str(tmp_repo))
        list(ingestor.scan(incremental=True))

        (tmp_repo / "new_file.py").write_text("def new(): pass\n", encoding="utf-8")
        second = list(ingestor.scan(incremental=True))

        rel_paths = {f.rel_path for f in second}
        assert "new_file.py" in rel_paths, "New file should be detected"


class TestScanSingleFile:
    """Tests for scan_single_file()."""

    def test_single_file_returns_codefile(self, tmp_repo: Path):
        """scan_single_file should return a valid CodeFile."""
        ingestor = MeshIngestor(str(tmp_repo))
        result = ingestor.scan_single_file(str(tmp_repo / "main.py"))

        assert result is not None, "Should return a CodeFile"
        assert isinstance(result, CodeFile)
        assert result.rel_path == "main.py"
        assert result.language == "python"
        assert result.size_bytes > 0
        assert result.line_count > 0
        assert len(result.file_hash) == 64  # SHA-256 hex length

    def test_single_file_nonexistent_returns_none(self, tmp_repo: Path):
        """Non-existent file should return None, not raise."""
        ingestor = MeshIngestor(str(tmp_repo))
        result = ingestor.scan_single_file(str(tmp_repo / "ghost.py"))
        assert result is None

    def test_single_file_unchanged_returns_none(self, tmp_repo: Path):
        """Second call without changes should return None (hash unchanged)."""
        ingestor = MeshIngestor(str(tmp_repo))
        first = ingestor.scan_single_file(str(tmp_repo / "main.py"))
        assert first is not None

        second = ingestor.scan_single_file(str(tmp_repo / "main.py"))
        assert second is None, "Unchanged file should return None"

    def test_single_file_outside_repo_returns_none(self, tmp_repo: Path, tmp_path: Path):
        """File outside repo root should return None."""
        outside = tmp_path / "outside.py"
        outside.write_text("print('outside')\n", encoding="utf-8")

        ingestor = MeshIngestor(str(tmp_repo))
        result = ingestor.scan_single_file(str(outside))
        assert result is None


class TestDeletedPaths:
    """Tests for get_deleted_paths()."""

    def test_no_deleted_files(self, tmp_repo: Path):
        """When no files are deleted, should return empty list."""
        ingestor = MeshIngestor(str(tmp_repo))
        list(ingestor.scan(incremental=False))
        deleted = ingestor.get_deleted_paths()
        assert deleted == []

    def test_detects_deleted_file(self, tmp_repo: Path):
        """Should detect files that disappeared since last scan."""
        ingestor = MeshIngestor(str(tmp_repo))
        list(ingestor.scan(incremental=False))

        # Delete main.py
        (tmp_repo / "main.py").unlink()
        deleted = ingestor.get_deleted_paths()
        assert "main.py" in deleted, "Deleted file should be detected"


class TestCodeFileDataclass:
    """Tests for CodeFile field correctness."""

    def test_language_detection(self, tmp_repo: Path):
        """Language should be correctly detected from extension."""
        ingestor = MeshIngestor(str(tmp_repo))
        files = {f.rel_path: f for f in ingestor.scan(incremental=False)}

        assert files["main.py"].language == "python"
        assert files["src/auth.js"].language == "javascript"
        assert files["src/server.ts"].language == "typescript"
        assert files["docs/notes.md"].language == "markdown"

    def test_content_is_actual_file_content(self, tmp_repo: Path):
        """CodeFile.content should contain the actual file content."""
        ingestor = MeshIngestor(str(tmp_repo))
        files = {f.rel_path: f for f in ingestor.scan(incremental=False)}

        actual_content = (tmp_repo / "main.py").read_text(encoding="utf-8")
        assert files["main.py"].content == actual_content

    def test_abs_path_is_absolute(self, tmp_repo: Path):
        """CodeFile.abs_path should be an absolute path."""
        ingestor = MeshIngestor(str(tmp_repo))
        for f in ingestor.scan(incremental=False):
            assert Path(f.abs_path).is_absolute(), f"abs_path not absolute: {f.abs_path}"

    def test_is_binary_property(self):
        """is_binary should detect null bytes in content."""
        normal = CodeFile(
            rel_path="a.py", abs_path="/a.py", content="print('hi')",
            file_hash="abc", language="python", size_bytes=10, line_count=1,
        )
        assert normal.is_binary is False

        binary = CodeFile(
            rel_path="b.bin", abs_path="/b.bin", content="data\x00here",
            file_hash="def", language="text", size_bytes=9, line_count=1,
        )
        assert binary.is_binary is True


class TestExtraPruning:
    """Tests for custom prune dirs/files."""

    def test_extra_prune_dirs(self, tmp_path: Path):
        """Custom prune dirs should be excluded."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "keep.py").write_text("x = 1\n", encoding="utf-8")
        custom_dir = repo / "my_cache"
        custom_dir.mkdir()
        (custom_dir / "cached.py").write_text("y = 2\n", encoding="utf-8")

        ingestor = MeshIngestor(str(repo), extra_prune_dirs={"my_cache"})
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}

        assert "keep.py" in rel_paths
        assert "my_cache/cached.py" not in rel_paths

    def test_extra_prune_files(self, tmp_path: Path):
        """Custom prune filenames should be excluded."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "keep.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "secret.key").write_text("supersecret\n", encoding="utf-8")

        ingestor = MeshIngestor(str(repo), extra_prune_files={"secret.key"})
        files = list(ingestor.scan(incremental=False))
        rel_paths = {f.rel_path for f in files}

        assert "keep.py" in rel_paths
        assert "secret.key" not in rel_paths
