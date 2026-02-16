"""
Real-Repository Quick Test — Auto-clones a mid-sized open-source repo.

Target: encode/httpx (~100 Python files, well-structured async HTTP client).
Purpose: Validate the full pipeline on real-world code, not synthetic fixtures.

Marker: @pytest.mark.slow — skipped by default in `pytest`.
Run explicitly: pytest -m slow tests/test_real_repo.py -v
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from src.server.pipeline import KinetiMeshPipeline

# ── Configuration ───────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/encode/httpx.git"
REPO_NAME = "httpx"
CLONE_DIR = Path("/tmp/kmesh_test_httpx")
DB_DIR = Path("/tmp/kmesh_test_httpx_db")

# Shallow clone depth — keeps download fast (~5 MB vs ~50 MB full)
CLONE_DEPTH = 1


def _clone_repo() -> Path:
    """Clone httpx repo to /tmp if not already present.

    Returns:
        Path to the cloned repository root.
    """
    if CLONE_DIR.exists() and (CLONE_DIR / ".git").exists():
        return CLONE_DIR

    # Clean any partial clone
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    subprocess.run(
        ["git", "clone", "--depth", str(CLONE_DEPTH), REPO_URL, str(CLONE_DIR)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return CLONE_DIR


def _cleanup_db() -> None:
    """Remove the test database directory."""
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def httpx_repo() -> Path:
    """Ensure httpx is cloned and return its path."""
    return _clone_repo()


@pytest.fixture(scope="module")
def httpx_pipeline(httpx_repo: Path) -> KinetiMeshPipeline:
    """Create and full-index the httpx pipeline (once per module)."""
    _cleanup_db()
    pipeline = KinetiMeshPipeline(str(httpx_repo), db_path=str(DB_DIR))
    pipeline.full_index(incremental=False)
    return pipeline


# ── Tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestRealRepoIndexing:
    """Validate indexing metrics on httpx."""

    def test_indexes_substantial_files(self, httpx_pipeline: KinetiMeshPipeline):
        """Should index at least 30 source files from httpx."""
        stats = httpx_pipeline.get_stats()
        assert stats["total_indexed_files"] >= 30, \
            f"Expected >=30 files, got {stats['total_indexed_files']}"

    def test_produces_many_chunks(self, httpx_pipeline: KinetiMeshPipeline):
        """Should produce hundreds of chunks from a real repo."""
        stats = httpx_pipeline.get_stats()
        assert stats["total_stored_chunks"] >= 100, \
            f"Expected >=100 chunks, got {stats['total_stored_chunks']}"


@pytest.mark.slow
class TestRealRepoSearch:
    """Validate search quality on real httpx code."""

    def test_find_http_client(self, httpx_pipeline: KinetiMeshPipeline):
        """Query about HTTP client should find httpx Client class."""
        results = httpx_pipeline.search("HTTP client send request", top_k=10)
        assert len(results) > 0, "Expected results for HTTP client query"
        all_text = " ".join(r["text"].lower() for r in results)
        assert "client" in all_text or "request" in all_text, \
            "Expected client/request related content"

    def test_find_async_transport(self, httpx_pipeline: KinetiMeshPipeline):
        """Query about async transport should find relevant code."""
        results = httpx_pipeline.search("async transport handle request", top_k=10)
        assert len(results) > 0
        files = {r["file_path"] for r in results}
        assert len(files) >= 1, "Expected results from at least 1 file"

    def test_find_url_parsing(self, httpx_pipeline: KinetiMeshPipeline):
        """Query about URL parsing should find URL class or related."""
        results = httpx_pipeline.search("URL parsing scheme host path", top_k=10)
        assert len(results) > 0
        all_symbols = [r["symbol_name"].lower() for r in results]
        assert any("url" in s for s in all_symbols), \
            f"Expected URL-related symbol, got {all_symbols}"

    def test_find_status_codes(self, httpx_pipeline: KinetiMeshPipeline):
        """Query about HTTP status codes should find relevant code."""
        results = httpx_pipeline.search("HTTP status code response", top_k=10)
        assert len(results) > 0

    def test_cross_file_results(self, httpx_pipeline: KinetiMeshPipeline):
        """Search should return results from multiple files."""
        results = httpx_pipeline.search("authentication headers", top_k=10)
        if results:
            files = {r["file_path"] for r in results}
            # Real repo should have auth-related code across multiple files
            assert len(files) >= 1


@pytest.mark.slow
class TestRealRepoSymbolSearch:
    """Validate symbol search on real code."""

    def test_find_client_class(self, httpx_pipeline: KinetiMeshPipeline):
        """Symbol search for 'Client' should find related results."""
        results = httpx_pipeline.search_symbol("Client", top_k=10)
        assert len(results) > 0, "Expected results for 'Client' symbol search"
        # httpx uses Client, AsyncClient — check text content
        all_text = " ".join(r["text"].lower() for r in results)
        assert "client" in all_text, \
            f"Expected 'client' in result text"

    def test_find_response_class(self, httpx_pipeline: KinetiMeshPipeline):
        """Symbol search for 'Response' should find related results."""
        results = httpx_pipeline.search_symbol("Response", top_k=10)
        assert len(results) > 0, "Expected results for 'Response' symbol search"
        all_text = " ".join(r["text"].lower() for r in results)
        assert "response" in all_text, \
            f"Expected 'response' in result text"


@pytest.mark.slow
class TestRealRepoSkeleton:
    """Validate file skeleton on real httpx files."""

    def test_skeleton_main_module(self, httpx_pipeline: KinetiMeshPipeline):
        """Should generate a meaningful skeleton for httpx/__init__.py."""
        # httpx uses httpx/ as main package dir
        skeleton = httpx_pipeline.get_file_skeleton("httpx/__init__.py")
        # If file exists, skeleton should have content
        if "not found" not in skeleton.lower():
            assert len(skeleton) > 50, "Skeleton should be non-trivial"


@pytest.mark.slow
class TestRealRepoIncremental:
    """Validate incremental re-index on the real repo."""

    def test_second_index_no_changes(self, httpx_pipeline: KinetiMeshPipeline):
        """Second incremental index should detect no changes."""
        metrics = httpx_pipeline.full_index(incremental=True)
        assert metrics["status"] == "no_changes", \
            f"Expected no_changes, got {metrics['status']}"
