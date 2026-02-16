"""
Tests for the VectorStore (src/db/store.py).

Covers:
    - Embedding and indexing code chunks
    - Vector search correctness (relevant results returned)
    - Full-text search (FTS) correctness
    - Hybrid search pipeline
    - File-level deletion
    - Index statistics
    - Store clear/rebuild
    - Edge cases: empty queries, no results, duplicate chunks
    - Result field validation (not just key existence)
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.parser.chunker import CodeChunk, ChunkType, parse_file
from src.db.store import VectorStore, embed_texts, embed_query
from tests.conftest import SAMPLE_PYTHON, SAMPLE_JAVASCRIPT


def _make_chunks(file_path: str = "test.py", language: str = "python") -> List[CodeChunk]:
    """Helper to create known chunks from sample Python code."""
    return parse_file(SAMPLE_PYTHON, language, file_path)


def _make_simple_chunk(
    text: str,
    symbol: str,
    file_path: str = "test.py",
    chunk_type: ChunkType = ChunkType.SIGNATURE,
) -> CodeChunk:
    """Create a single minimal chunk for testing."""
    return CodeChunk(
        text=text,
        chunk_type=chunk_type,
        symbol_name=symbol,
        start_line=1,
        end_line=1,
        language="python",
        file_path=file_path,
        priority=2.0 if chunk_type == ChunkType.SIGNATURE else 1.0,
    )


class TestEmbedding:
    """Tests for embedding functions."""

    def test_embed_texts_shape(self):
        """embed_texts should return (n, 384) array."""
        texts = ["hello world", "def foo(): pass", "class Bar:"]
        vectors = embed_texts(texts, show_progress=False)
        assert vectors.shape == (3, 384), f"Expected (3, 384), got {vectors.shape}"

    def test_embed_texts_normalized(self):
        """Vectors should be L2-normalized (unit vectors)."""
        texts = ["test normalization"]
        vectors = embed_texts(texts, show_progress=False)
        norm = np.linalg.norm(vectors[0])
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"

    def test_embed_query_shape(self):
        """embed_query should return a (384,) vector."""
        vec = embed_query("search for something")
        assert vec.shape == (384,), f"Expected (384,), got {vec.shape}"

    def test_embed_empty_list(self):
        """Embedding empty list should return (0, 384) or handle gracefully."""
        vectors = embed_texts([], show_progress=False)
        assert vectors.shape[0] == 0

    def test_similar_texts_closer_than_different(self):
        """Semantically similar texts should have higher cosine similarity."""
        vecs = embed_texts([
            "def add(a, b): return a + b",
            "function sum(x, y) { return x + y; }",
            "the weather is sunny today",
        ], show_progress=False)
        # cosine sim of normalized vectors = dot product
        sim_code = np.dot(vecs[0], vecs[1])
        sim_unrelated = np.dot(vecs[0], vecs[2])
        assert sim_code > sim_unrelated, \
            f"Similar code ({sim_code:.3f}) should be closer than unrelated ({sim_unrelated:.3f})"


class TestVectorStoreIndexing:
    """Tests for chunk indexing into the vector store."""

    def test_index_chunks_stores_data(self, tmp_db_path: str):
        """Indexing chunks should populate the store."""
        store = VectorStore(db_path=tmp_db_path)
        chunks = _make_chunks()
        assert len(chunks) > 0, "Sample code should produce chunks"

        metrics = store.index_chunks(chunks, full_rebuild=True)

        assert metrics["chunks_indexed"] == len(chunks)
        assert metrics["embed_time_ms"] > 0
        stats = store.get_stats()
        assert stats["total_chunks"] == len(chunks), \
            f"Expected {len(chunks)} stored, got {stats['total_chunks']}"

    def test_index_returns_correct_metrics(self, tmp_db_path: str):
        """index_chunks should return meaningful metric values."""
        store = VectorStore(db_path=tmp_db_path)
        chunks = _make_chunks()
        metrics = store.index_chunks(chunks, full_rebuild=True)

        assert "embed_time_ms" in metrics
        assert "store_time_ms" in metrics
        assert "chunks_indexed" in metrics
        assert "files_indexed" in metrics
        assert metrics["files_indexed"] >= 1
        assert metrics["embed_time_ms"] >= 0
        assert metrics["store_time_ms"] >= 0

    def test_index_empty_chunks(self, tmp_db_path: str):
        """Indexing zero chunks should return zero metrics, not crash."""
        store = VectorStore(db_path=tmp_db_path)
        metrics = store.index_chunks([])
        assert metrics["chunks_indexed"] == 0

    def test_incremental_index_replaces_file(self, tmp_db_path: str):
        """Re-indexing chunks for a file should replace, not duplicate."""
        store = VectorStore(db_path=tmp_db_path)
        chunks_v1 = [_make_simple_chunk("def old(): pass", "old", "file_a.py")]
        store.index_chunks(chunks_v1, full_rebuild=True)
        assert store.get_stats()["total_chunks"] == 1

        chunks_v2 = [
            _make_simple_chunk("def new_a(): pass", "new_a", "file_a.py"),
            _make_simple_chunk("def new_b(): pass", "new_b", "file_a.py"),
        ]
        store.index_chunks(chunks_v2, full_rebuild=False)
        # Should have 2 chunks (old one replaced), not 3
        assert store.get_stats()["total_chunks"] == 2

    def test_full_rebuild_replaces_everything(self, tmp_db_path: str):
        """full_rebuild=True should drop all existing data."""
        store = VectorStore(db_path=tmp_db_path)
        store.index_chunks([_make_simple_chunk("def a(): pass", "a")], full_rebuild=True)
        assert store.get_stats()["total_chunks"] == 1

        store.index_chunks([_make_simple_chunk("def b(): pass", "b")], full_rebuild=True)
        # Full rebuild drops everything, then adds only new
        assert store.get_stats()["total_chunks"] == 1


class TestVectorSearch:
    """Tests for vector similarity search."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_db_path: str):
        """Pre-populate store with sample Python + JavaScript chunks."""
        self.store = VectorStore(db_path=tmp_db_path)
        py_chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        js_chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        self.store.index_chunks(py_chunks + js_chunks, full_rebuild=True)

    def test_vector_search_returns_results(self):
        """Searching for known code should return results."""
        results = self.store.search_vector("fibonacci recursive function", top_k=5)
        assert len(results) > 0, "Expected search results for 'fibonacci'"

    def test_vector_search_result_fields(self):
        """Each result should have all expected fields with correct types."""
        results = self.store.search_vector("calculator", top_k=3)
        assert len(results) > 0

        r = results[0]
        required_keys = {"id", "text", "file_path", "symbol_name", "chunk_type",
                         "language", "start_line", "end_line", "priority", "score"}
        assert required_keys.issubset(r.keys()), \
            f"Missing keys: {required_keys - r.keys()}"

        assert isinstance(r["text"], str) and len(r["text"]) > 0
        assert isinstance(r["start_line"], int) and r["start_line"] >= 1
        assert isinstance(r["score"], float)

    def test_vector_search_relevance(self):
        """Top result for 'fibonacci' should reference fibonacci function."""
        results = self.store.search_vector("fibonacci number calculation", top_k=3)
        top_symbols = [r["symbol_name"] for r in results[:3]]
        assert any("fibonacci" in s.lower() for s in top_symbols), \
            f"Expected fibonacci in top 3, got {top_symbols}"

    def test_vector_search_cross_language(self):
        """Search for 'authentication' should find JS auth code."""
        results = self.store.search_vector("user authentication login", top_k=5)
        assert len(results) > 0
        languages = {r["language"] for r in results}
        assert "javascript" in languages, \
            f"Expected JS results for auth query, got languages: {languages}"

    def test_vector_search_top_k_respected(self):
        """Should return at most top_k results."""
        results = self.store.search_vector("code", top_k=2)
        assert len(results) <= 2

    def test_vector_search_empty_query(self):
        """Empty query should not crash."""
        results = self.store.search_vector("", top_k=5)
        assert isinstance(results, list)

    def test_vector_search_no_match_query(self):
        """Query with no semantic match should still return results (nearest neighbors)."""
        results = self.store.search_vector(
            "quantum entanglement photon interference", top_k=3
        )
        # Vector search always returns nearest neighbors
        assert isinstance(results, list)


class TestFTSSearch:
    """Tests for full-text (keyword) search."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_db_path: str):
        """Pre-populate store with sample chunks."""
        self.store = VectorStore(db_path=tmp_db_path)
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        self.store.index_chunks(chunks, full_rebuild=True)

    def test_fts_finds_exact_keyword(self):
        """FTS should find chunks containing exact keywords."""
        results = self.store.search_fts("fibonacci", top_k=5)
        assert len(results) > 0, "Expected FTS results for 'fibonacci'"
        assert any("fibonacci" in r["text"].lower() for r in results)

    def test_fts_no_match(self):
        """FTS with non-existent keyword should return empty list."""
        results = self.store.search_fts("xyznonexistentterm123", top_k=5)
        assert results == []


class TestHybridSearch:
    """Tests for hybrid (vector + FTS + reranker) search."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_db_path: str):
        """Pre-populate store."""
        self.store = VectorStore(db_path=tmp_db_path)
        py_chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        js_chunks = parse_file(SAMPLE_JAVASCRIPT, "javascript", "auth.js")
        self.store.index_chunks(py_chunks + js_chunks, full_rebuild=True)

    def test_hybrid_search_returns_results(self):
        """Hybrid search should combine vector and FTS results."""
        results = self.store.search_hybrid("divide by zero", top_k=5, use_reranker=False)
        assert len(results) > 0

    def test_hybrid_search_with_reranker(self):
        """Hybrid search with re-ranking should not crash."""
        results = self.store.search_hybrid("fibonacci", top_k=3, use_reranker=True)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_deduplicates(self):
        """Results should not contain duplicate chunk IDs."""
        results = self.store.search_hybrid("calculator add", top_k=10, use_reranker=False)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


class TestSymbolSearch:
    """Tests for symbol-specific search."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_db_path: str):
        """Pre-populate store."""
        self.store = VectorStore(db_path=tmp_db_path)
        chunks = parse_file(SAMPLE_PYTHON, "python", "main.py")
        self.store.index_chunks(chunks, full_rebuild=True)

    def test_symbol_search_finds_definition(self):
        """Searching for 'Calculator' should find the class definition."""
        results = self.store.search_symbol("Calculator", top_k=5)
        assert len(results) > 0
        types = {r["chunk_type"] for r in results}
        assert "class_definition" in types or "signature" in types, \
            f"Expected definition chunk, got types: {types}"

    def test_symbol_search_finds_method(self):
        """Searching for a method name should find it."""
        results = self.store.search_symbol("fibonacci", top_k=5)
        assert len(results) > 0
        assert any("fibonacci" in r["symbol_name"].lower() for r in results)


class TestDeleteFile:
    """Tests for file deletion from the store."""

    def test_delete_removes_file_chunks(self, tmp_db_path: str):
        """Deleting a file should remove all its chunks from the store."""
        store = VectorStore(db_path=tmp_db_path)
        chunks = parse_file(SAMPLE_PYTHON, "python", "to_delete.py")
        store.index_chunks(chunks, full_rebuild=True)

        count_before = store.get_stats()["total_chunks"]
        assert count_before > 0

        store.delete_file("to_delete.py")
        count_after = store.get_stats()["total_chunks"]
        assert count_after == 0, f"Expected 0 chunks after delete, got {count_after}"

    def test_delete_nonexistent_file_no_crash(self, tmp_db_path: str):
        """Deleting a file that doesn't exist should not raise."""
        store = VectorStore(db_path=tmp_db_path)
        chunks = _make_chunks()
        store.index_chunks(chunks, full_rebuild=True)

        # Should not crash
        store.delete_file("nonexistent_file.py")
        assert store.get_stats()["total_chunks"] > 0  # Other data unchanged

    def test_delete_only_affects_target_file(self, tmp_db_path: str):
        """Deleting file_a should not affect file_b chunks."""
        store = VectorStore(db_path=tmp_db_path)
        chunks_a = [_make_simple_chunk("def a(): pass", "a", "file_a.py")]
        chunks_b = [_make_simple_chunk("def b(): pass", "b", "file_b.py")]
        store.index_chunks(chunks_a + chunks_b, full_rebuild=True)
        assert store.get_stats()["total_chunks"] == 2

        store.delete_file("file_a.py")
        assert store.get_stats()["total_chunks"] == 1


class TestStoreClear:
    """Tests for store.clear()."""

    def test_clear_removes_all_data(self, tmp_db_path: str):
        """clear() should drop all chunks."""
        store = VectorStore(db_path=tmp_db_path)
        store.index_chunks(_make_chunks(), full_rebuild=True)
        assert store.get_stats()["total_chunks"] > 0

        store.clear()
        assert store.get_stats()["total_chunks"] == 0

    def test_clear_then_reindex(self, tmp_db_path: str):
        """Should be able to re-index after clear()."""
        store = VectorStore(db_path=tmp_db_path)
        store.index_chunks(_make_chunks(), full_rebuild=True)
        store.clear()

        # Re-index
        chunks = _make_chunks()
        store.index_chunks(chunks, full_rebuild=True)
        assert store.get_stats()["total_chunks"] == len(chunks)


class TestStoreStats:
    """Tests for get_stats()."""

    def test_stats_keys(self, tmp_db_path: str):
        """Stats should include expected keys."""
        store = VectorStore(db_path=tmp_db_path)
        stats = store.get_stats()
        assert "total_chunks" in stats
        assert "db_path" in stats

    def test_stats_reflect_indexed_count(self, tmp_db_path: str):
        """total_chunks should match the number of indexed chunks."""
        store = VectorStore(db_path=tmp_db_path)
        chunks = _make_chunks()
        store.index_chunks(chunks, full_rebuild=True)
        assert store.get_stats()["total_chunks"] == len(chunks)
