"""
Stress Test — FastAPI (~2800 files, multi-language).

Clones the full FastAPI repository and validates the pipeline handles
a large, real-world codebase without errors, memory issues, or
unacceptable latency.

Marker: @pytest.mark.stress — skipped by default.
Run explicitly: pytest -m stress tests/test_stress.py -v
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from src.server.pipeline import KinetiMeshPipeline

# ── Configuration ───────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/fastapi/fastapi.git"
REPO_NAME = "fastapi"
CLONE_DIR = Path("/tmp/kmesh_stress_fastapi")
DB_DIR = Path("/tmp/kmesh_stress_fastapi_db")
CLONE_DEPTH = 1

# Performance thresholds (generous for CI variance)
MAX_INDEX_TIME_S = 600       # Full index < 10 minutes (includes model cold-start)
MAX_SEARCH_TIME_MS = 500     # Single search < 500ms
MIN_FILES_INDEXED = 200      # FastAPI has 200+ indexable source files
MIN_CHUNKS_PRODUCED = 1000   # Should produce 1000+ chunks


def _clone_repo() -> Path:
    """Clone FastAPI repo to /tmp if not already present."""
    if CLONE_DIR.exists() and (CLONE_DIR / ".git").exists():
        return CLONE_DIR

    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    subprocess.run(
        ["git", "clone", "--depth", str(CLONE_DEPTH), REPO_URL, str(CLONE_DIR)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return CLONE_DIR


def _cleanup_db() -> None:
    """Remove the stress test database."""
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fastapi_repo() -> Path:
    """Ensure FastAPI is cloned."""
    return _clone_repo()


@pytest.fixture(scope="module")
def fastapi_pipeline(fastapi_repo: Path) -> KinetiMeshPipeline:
    """Full-index FastAPI (once per module)."""
    _cleanup_db()
    pipeline = KinetiMeshPipeline(str(fastapi_repo), db_path=str(DB_DIR))
    start = time.perf_counter()
    pipeline.full_index(incremental=False)
    elapsed = time.perf_counter() - start
    print(f"\n[STRESS] FastAPI full index: {elapsed:.2f}s")
    return pipeline


# ── Tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.stress
class TestStressIndexing:
    """Validate indexing performance on FastAPI."""

    def test_indexes_many_files(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should index 200+ source files from FastAPI."""
        stats = fastapi_pipeline.get_stats()
        assert stats["total_indexed_files"] >= MIN_FILES_INDEXED, \
            f"Expected >={MIN_FILES_INDEXED} files, got {stats['total_indexed_files']}"

    def test_produces_many_chunks(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should produce 1000+ chunks."""
        stats = fastapi_pipeline.get_stats()
        assert stats["total_stored_chunks"] >= MIN_CHUNKS_PRODUCED, \
            f"Expected >={MIN_CHUNKS_PRODUCED} chunks, got {stats['total_stored_chunks']}"

    def test_index_time_acceptable(self, fastapi_pipeline: KinetiMeshPipeline):
        """Full index should complete within threshold."""
        t = fastapi_pipeline.last_full_index_time
        assert t < MAX_INDEX_TIME_S, \
            f"Full index took {t:.1f}s, threshold is {MAX_INDEX_TIME_S}s"


@pytest.mark.stress
class TestStressSearchLatency:
    """Validate search latency under load."""

    QUERIES = [
        "dependency injection",
        "OAuth2 authentication middleware",
        "request validation Pydantic model",
        "WebSocket connection handler",
        "background task scheduling",
        "CORS middleware configuration",
        "API route decorator path operation",
        "exception handler HTTP error",
    ]

    def test_all_queries_return_results(self, fastapi_pipeline: KinetiMeshPipeline):
        """All semantic queries should return results on a large codebase."""
        for query in self.QUERIES:
            results = fastapi_pipeline.search(query, top_k=5, use_reranker=False)
            assert len(results) > 0, f"No results for: '{query}'"

    def test_search_latency(self, fastapi_pipeline: KinetiMeshPipeline):
        """Each search should complete under the latency threshold."""
        for query in self.QUERIES:
            start = time.perf_counter()
            fastapi_pipeline.search(query, top_k=5, use_reranker=False)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < MAX_SEARCH_TIME_MS, \
                f"Search '{query}' took {elapsed_ms:.0f}ms, limit is {MAX_SEARCH_TIME_MS}ms"

    def test_avg_search_latency(self, fastapi_pipeline: KinetiMeshPipeline):
        """Average search latency across all queries should be reasonable."""
        times = []
        for query in self.QUERIES:
            start = time.perf_counter()
            fastapi_pipeline.search(query, top_k=5, use_reranker=False)
            times.append((time.perf_counter() - start) * 1000)

        avg = sum(times) / len(times)
        print(f"\n[STRESS] Avg search latency: {avg:.1f}ms (n={len(times)})")
        assert avg < MAX_SEARCH_TIME_MS / 2, \
            f"Avg search {avg:.0f}ms exceeds {MAX_SEARCH_TIME_MS // 2}ms"


@pytest.mark.stress
class TestStressSearchQuality:
    """Validate search relevance on FastAPI code."""

    def test_find_dependency_injection(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should find FastAPI's Depends system."""
        results = fastapi_pipeline.search("dependency injection Depends", top_k=10)
        assert len(results) > 0
        all_text = " ".join(r["text"].lower() for r in results)
        assert "depends" in all_text or "dependency" in all_text

    def test_find_router(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should find APIRouter class."""
        results = fastapi_pipeline.search_symbol("APIRouter", top_k=5)
        assert len(results) > 0
        assert any("APIRouter" in r["symbol_name"] for r in results)

    def test_find_fastapi_class(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should find the main FastAPI application class."""
        results = fastapi_pipeline.search_symbol("FastAPI", top_k=5)
        assert len(results) > 0

    def test_find_websocket(self, fastapi_pipeline: KinetiMeshPipeline):
        """Should find WebSocket handling code."""
        results = fastapi_pipeline.search("WebSocket connection accept send", top_k=10)
        assert len(results) > 0


@pytest.mark.stress
class TestStressIncremental:
    """Validate incremental re-index on a large repo."""

    def test_incremental_no_changes(self, fastapi_pipeline: KinetiMeshPipeline):
        """Second incremental should be near-instant (no changes)."""
        start = time.perf_counter()
        metrics = fastapi_pipeline.full_index(incremental=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert metrics["status"] == "no_changes"
        # No-change incremental should be < 3s (just scan, no embed)
        assert elapsed_ms < 3000, \
            f"No-change incremental took {elapsed_ms:.0f}ms, should be <3000ms"
        print(f"\n[STRESS] No-change incremental: {elapsed_ms:.0f}ms")
