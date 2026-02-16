"""
Tests for the KinetiMeshPipeline orchestrator (src/server/pipeline.py).

Covers:
    - Full index on a test repo
    - Incremental index (only changed files processed)
    - Single-file index
    - File deletion handling
    - Search through pipeline
    - Symbol search through pipeline
    - File skeleton through pipeline
    - Pipeline stats
    - Watcher lifecycle (start/stop)
    - Isolation: tests use tmp dirs, never touch production data
"""

import time
from pathlib import Path
from typing import Dict, Any

import pytest

from src.server.pipeline import KinetiMeshPipeline
from tests.conftest import SAMPLE_PYTHON, SAMPLE_JAVASCRIPT


class TestFullIndex:
    """Tests for full_index()."""

    def test_full_index_returns_metrics(self, tmp_repo: Path, tmp_db_path: str):
        """full_index should return a dict with timing and count metrics."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        metrics = pipeline.full_index(incremental=False)

        assert metrics["status"] == "indexed"
        assert metrics["files_scanned"] > 0, "Should scan files"
        assert metrics["chunks_parsed"] > 0, "Should produce chunks"
        assert metrics["total_time_ms"] > 0
        assert metrics["scan_time_ms"] >= 0
        assert metrics["parse_time_ms"] >= 0

    def test_full_index_populates_store(self, tmp_repo: Path, tmp_db_path: str):
        """After full_index, the store should contain chunks."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)

        stats = pipeline.get_stats()
        assert stats["total_stored_chunks"] > 0, "Store should have chunks"
        assert stats["total_indexed_files"] > 0

    def test_full_index_empty_repo(self, empty_dir: Path, tmp_db_path: str):
        """Indexing an empty repo should succeed with no_changes status."""
        pipeline = KinetiMeshPipeline(str(empty_dir), db_path=tmp_db_path)
        metrics = pipeline.full_index(incremental=False)
        assert metrics["status"] == "no_changes"
        assert metrics["files_scanned"] == 0


class TestIncrementalIndex:
    """Tests for incremental indexing behavior."""

    def test_second_index_no_changes(self, tmp_repo: Path, tmp_db_path: str):
        """Second incremental index without changes should report no_changes."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=True)
        metrics = pipeline.full_index(incremental=True)
        assert metrics["status"] == "no_changes"

    def test_incremental_detects_modification(self, tmp_repo: Path, tmp_db_path: str):
        """Modifying a file triggers re-indexing on next incremental pass."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=True)

        # Modify a file
        (tmp_repo / "main.py").write_text("# modified\nprint('new')\n", encoding="utf-8")
        metrics = pipeline.full_index(incremental=True)

        assert metrics["status"] == "indexed"
        assert metrics["files_scanned"] >= 1


class TestSingleFileIndex:
    """Tests for index_single_file()."""

    def test_index_single_file(self, tmp_repo: Path, tmp_db_path: str):
        """Single file indexing should work independently."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        metrics = pipeline.index_single_file(str(tmp_repo / "main.py"))

        assert metrics["status"] == "indexed"
        assert metrics["chunks"] > 0

    def test_index_unchanged_file(self, tmp_repo: Path, tmp_db_path: str):
        """Re-indexing an unchanged file should report 'unchanged'."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.index_single_file(str(tmp_repo / "main.py"))
        metrics = pipeline.index_single_file(str(tmp_repo / "main.py"))
        assert metrics["status"] == "unchanged"

    def test_index_nonexistent_file(self, tmp_repo: Path, tmp_db_path: str):
        """Indexing a nonexistent file should report 'unchanged', not crash."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        metrics = pipeline.index_single_file(str(tmp_repo / "ghost.py"))
        assert metrics["status"] == "unchanged"


class TestFileDelete:
    """Tests for handle_file_delete()."""

    def test_delete_removes_from_index(self, tmp_repo: Path, tmp_db_path: str):
        """Deleting a file should remove its chunks from the store."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)

        chunks_before = pipeline.get_stats()["total_stored_chunks"]
        assert chunks_before > 0

        # Delete main.py from index
        pipeline.handle_file_delete(str(tmp_repo / "main.py"))

        chunks_after = pipeline.get_stats()["total_stored_chunks"]
        assert chunks_after < chunks_before, \
            f"Chunks should decrease after delete: {chunks_before} -> {chunks_after}"

    def test_delete_nonexistent_file_no_crash(self, tmp_repo: Path, tmp_db_path: str):
        """Deleting a file that was never indexed should not crash."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)
        # Should not raise
        pipeline.handle_file_delete(str(tmp_repo / "never_existed.py"))


class TestPipelineSearch:
    """Tests for search through the pipeline."""

    @pytest.fixture(autouse=True)
    def _setup_pipeline(self, tmp_repo: Path, tmp_db_path: str):
        """Pre-index the test repo."""
        self.pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        self.pipeline.full_index(incremental=False)

    def test_search_returns_results(self):
        """Searching for known content should return results."""
        results = self.pipeline.search("fibonacci recursive", top_k=5)
        assert len(results) > 0, "Expected search results"

    def test_search_result_content(self):
        """Search results should contain actual code text."""
        results = self.pipeline.search("calculator add numbers", top_k=3)
        assert len(results) > 0
        # At least one result should contain calculator-related text
        all_text = " ".join(r["text"].lower() for r in results)
        assert "add" in all_text or "calculator" in all_text, \
            "Expected calculator/add related content in results"

    def test_search_empty_query(self):
        """Empty query should not crash."""
        results = self.pipeline.search("", top_k=5)
        assert isinstance(results, list)


class TestSymbolSearch:
    """Tests for symbol search through pipeline."""

    @pytest.fixture(autouse=True)
    def _setup_pipeline(self, tmp_repo: Path, tmp_db_path: str):
        """Pre-index the test repo."""
        self.pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        self.pipeline.full_index(incremental=False)

    def test_symbol_search_finds_class(self):
        """Should find Calculator class definition."""
        results = self.pipeline.search_symbol("Calculator", top_k=5)
        assert len(results) > 0
        symbols = {r["symbol_name"] for r in results}
        assert any("Calculator" in s for s in symbols)

    def test_symbol_search_finds_function(self):
        """Should find fibonacci function."""
        results = self.pipeline.search_symbol("fibonacci", top_k=5)
        assert len(results) > 0


class TestFileSkeleton:
    """Tests for get_file_skeleton() through pipeline."""

    def test_skeleton_existing_file(self, tmp_repo: Path, tmp_db_path: str):
        """Should return skeleton for an existing file."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        skeleton = pipeline.get_file_skeleton("main.py")

        assert "Calculator" in skeleton
        assert "fibonacci" in skeleton
        assert isinstance(skeleton, str)
        assert len(skeleton) > 10

    def test_skeleton_nonexistent_file(self, tmp_repo: Path, tmp_db_path: str):
        """Should return error message for nonexistent file."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        result = pipeline.get_file_skeleton("does_not_exist.py")
        assert "not found" in result.lower() or "error" in result.lower()


class TestPipelineStats:
    """Tests for pipeline statistics."""

    def test_stats_keys(self, tmp_repo: Path, tmp_db_path: str):
        """Stats should contain expected keys."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)
        stats = pipeline.get_stats()

        expected_keys = {
            "repo_path", "total_indexed_files", "total_indexed_chunks",
            "tracked_files", "total_stored_chunks",
        }
        assert expected_keys.issubset(stats.keys()), \
            f"Missing keys: {expected_keys - stats.keys()}"

    def test_stats_values_after_index(self, tmp_repo: Path, tmp_db_path: str):
        """Stats values should be positive after indexing."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)
        stats = pipeline.get_stats()

        assert stats["total_indexed_files"] > 0
        assert stats["total_indexed_chunks"] > 0
        assert stats["total_stored_chunks"] > 0


class TestWatcherLifecycle:
    """Tests for file watcher start/stop."""

    def test_start_and_stop_watcher(self, tmp_repo: Path, tmp_db_path: str):
        """Watcher should start and stop without errors."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.full_index(incremental=False)

        assert not pipeline.get_stats()["watcher_active"]
        pipeline.start_watcher()
        assert pipeline.get_stats()["watcher_active"]
        pipeline.stop_watcher()
        assert not pipeline.get_stats()["watcher_active"]

    def test_double_start_noop(self, tmp_repo: Path, tmp_db_path: str):
        """Starting watcher twice should be idempotent."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.start_watcher()
        pipeline.start_watcher()  # should not crash or duplicate
        assert pipeline.get_stats()["watcher_active"]
        pipeline.stop_watcher()

    def test_stop_without_start(self, tmp_repo: Path, tmp_db_path: str):
        """Stopping watcher without starting should be safe."""
        pipeline = KinetiMeshPipeline(str(tmp_repo), db_path=tmp_db_path)
        pipeline.stop_watcher()  # should not crash
